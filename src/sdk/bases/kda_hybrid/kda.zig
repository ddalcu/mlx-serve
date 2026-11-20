//! A KDA layer (GatedDeltaNet with a per-channel gate) on one slot's recurrent state: GLM-5.3-Flash's (low-rank
//! gates, the row-joined decode projection, the fused one-dispatch decode and verify, the per-core prompt recurrence,
//! a bounded-sigmoid gate) and Ling 3.0's (full-rank gates, the softplus gate, the stock vector-gate kernels). A
//! verify captures every row's state and conv input for a partial accept's rollback.
//! copied_from: sdk/blocks/gdn/gated_delta_net.zig (`gatedDeltaNetProjected`, its arms a per-channel gate reaches),
//! sdk/blocks/gdn/gate.zig (`computeGdnGate`).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const compiled = @import("../blocks/compiled.zig");
const conv1d = @import("../blocks/conv1d.zig");
const kda_recurrence = @import("../../blocks/kda/recurrence.zig");
const kdaGateChain = @import("../../blocks/kda/gate.zig").kdaGateChain;
const gate_mod = @import("../../blocks/gdn/gate.zig");
const gdn_decode = @import("../../blocks/gdn/decode.zig");
const decode_fused = @import("../../blocks/gdn/decode_fused.zig");
const kernels = @import("../../blocks/gdn/kernels.zig");
const sliceQkvPart = @import("../../blocks/linear/fused_rows.zig").sliceQkvPart;
const trunk = @import("trunk.zig");

const Array = mlx.mlx_array;
const Model = trunk.Model;
const Kda = trunk.Kda;

/// One slot's recurrent state at a layer, with a verify's per-row capture; each set array is a reference of its own.
pub const Entry = struct {
    conv_state: Array = .{},
    ssm_state: Array = .{},
    spec_conv_input: Array = .{},
    spec_state_seq: Array = .{},
    initialized: bool = false,

    pub fn deinit(e: *Entry) void {
        inline for (.{ e.conv_state, e.ssm_state, e.spec_conv_input, e.spec_state_seq }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        e.* = .{};
    }
};

/// A low-rank gate's second stage as the fused KDA step reads it (a dense one at the pack's declared params).
pub fn low(m: *const Model, l: *const trunk.Linear) kda_recurrence.Low {
    const dense = l.s.ctx == null;
    return .{ .w = l.w, .s = l.s, .b = l.b, .bits = if (dense) m.c.quant_bits else l.bits, .gs = if (dense) m.c.quant_group_size else l.group_size };
}

fn free(a: Array) void {
    if (a.ctx != null) _ = mlx.mlx_array_free(a);
}

fn scalarOf(v: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !Array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

/// The q/k RMS norm's epsilon: GLM's reference normalizes with fla's l2norm (1e-6 on the SUM of squares).
fn qkNormEps(m: *const Model) f32 {
    return if (m.c.isGlm5()) 1e-6 / @as(f32, @floatFromInt(m.c.linear_key_head_dim)) else 1e-6;
}

/// `exp(-exp(A_log) * softplus(a + dt_bias))`, compiled where the load compiled it.
fn softplusGate(m: *const Model, A_log: Array, a: Array, dt_bias: Array) !Array {
    if (m.kda.gate_fn.ctx != null) {
        var g = ops.Graph.init(std.heap.c_allocator, m.s);
        defer g.deinit();
        if (try compiled.apply(&g, m.kda.gate_fn, &.{ A_log, a, dt_bias }, 1)) |out| {
            var r = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&r, out[0]);
            return r;
        }
    }
    return gate_mod.gdnGateChain(A_log, a, dt_bias, gate_mod.gdnChainGateDtype(mlx.mlx_array_dtype(a), m.ssmStateDtype()), m.s);
}

/// The layer over `x` `[B, S, hidden]` on `e`; `capture` keeps every row's state for rollback (a verify).
pub fn forward(m: *Model, x: Array, la: *const Kda, e: *Entry, batch: c_int, seq_len: c_int, is_prefill: bool, capture: bool) !Array {
    if (m.serial_rows and capture and batch == 1 and !la.in.built() and e.initialized and e.conv_state.ctx != null and e.ssm_state.ctx != null)
        return serialRows(m, x, la, e, seq_len);
    return forwardWith(m, x, la, e, batch, seq_len, is_prefill, capture, null);
}

/// A model plugin's own stages of a KDA layer's forward (`trunk.KdaConsts.stages`, set after load). Each runs where its
/// rows fall and takes the stock stage's place, or gives null where it does not serve the call, and the stock stage
/// runs. A stage that gives null has changed none of its inputs and no slot state; when it is asked, only the stock
/// stages before it have run (for recurrence and verify rows, the conv, which advances the conv window). Prompt rows
/// are one slot's 17 rows or more of a bounded gate, not captured and not on the per-core recurrence's arm; verify rows
/// are one slot's captured rows of a bounded gate.
pub const Stages = struct {
    /// Prompt rows with a 4-tap conv: the conv window and the scaled q, k and v heads in place of the conv, the q/k
    /// norms and the scales.
    prologue: ?*const fn (s: mlx.mlx_stream, in: PrologueIn) anyerror!?PrologueOut = null,
    /// Prompt rows on a GPU stream, after the prologue: the recurrence's output and final state. It makes its own gate,
    /// so the stock gate is not made where `serves` holds, and `run` then serves.
    recurrence: ?Recurrence = null,
    /// Verify rows: the gate in place of the composed gate chain.
    verify_gate: ?*const fn (s: mlx.mlx_stream, in: GateIn) anyerror!?Array = null,
    /// Verify rows on a GPU stream: the capture recurrence's output, final state and every row's state (what a partial
    /// accept's rollback reads) in place of the stock capture kernel.
    verify_capture: ?*const fn (s: mlx.mlx_stream, in: CaptureIn) anyerror!?CaptureOut = null,
    /// Verify rows with a 4-tap conv: the conv window, the scaled heads, beta and the capture's conv input in place of
    /// the composed conv, q/k norms, scales and beta sigmoid. Asked before the conv, so no slot state has changed then.
    verify_prework: ?*const fn (s: mlx.mlx_stream, in: PreworkIn) anyerror!?PreworkOut = null,
    /// Prompt rows and verify rows off the fused prework: the gated norm's flat output `[B, S, hv dv]` (the out
    /// projection's input) in place of the composed `rms_norm(y) * sigmoid(z)`.
    epilogue: ?*const fn (s: mlx.mlx_stream, in: EpilogueIn) anyerror!?Array = null,
};

/// The verify prework's operands, borrowed: as `PrologueIn`, with the beta projection `b` and the row count.
pub const PreworkIn = struct { qkv: Array, b: Array, conv_state: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, qk_eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int, rows: c_int };
/// New references the trunk owns: q, k `[1, S, hk, dk]`, v `[1, S, hv, dv]`, beta, the conv window after the last row
/// and the conv input the capture keeps (`Entry.spec_conv_input`).
pub const PreworkOut = struct { q: Array, k: Array, v: Array, beta: Array, window: Array, conv_input: Array };

/// The epilogue's operands, borrowed: the recurrence's y `[B, S, hv, dv]`, the z projection, the norm's weight and its
/// 0-dim epsilon; `verify`: the rows are a verify's.
pub const EpilogueIn = struct { y: Array, z: Array, norm_w: Array, eps: Array, hv: c_int, dv: c_int, batch: c_int, rows: c_int, verify: bool };

/// The prologue's operands, borrowed for the call: `[1, S, C]` q|k|v (`C = 2 hk dk + hv dv`), the conv window before
/// the first row (null: a cold slot), the depthwise conv's weight, the 0-dim q and k scales, the q/k norm's epsilon.
pub const PrologueIn = struct { qkv: Array, conv_state: ?Array, conv_w: Array, q_scale: Array, k_scale: Array, qk_eps: f32, hk: c_int, dk: c_int, hv: c_int, dv: c_int, taps: c_int };
/// New references the trunk owns: q, k `[1, S, hk, dk]`, v `[1, S, hv, dv]` and the conv window after the last row.
pub const PrologueOut = struct { q: Array, k: Array, v: Array, window: Array };

/// The recurrence's operands, borrowed: the scaled heads, the gate's input `a`, beta, the gate's per-head `A_log` and
/// `dt_bias` in f32 (as the loader keeps them for GLM; empty where it keeps none), its lower bound, and the state
/// before the first row.
pub const RecurIn = struct { q: Array, k: Array, v: Array, a: Array, beta: Array, a_log_h: Array, dt_f32: Array, lower_bound: f32, state: Array };
/// New references the trunk owns: y `[1, S, hv, dv]` and the state after the last row.
pub const RecurOut = struct { y: Array, state: Array };
/// `serves` sees the stream, the geometry, the dtypes and the f32 gate operands (empty where the loader keeps none) and
/// holds where `run` serves these rows.
pub const Recurrence = struct {
    serves: *const fn (s: mlx.mlx_stream, hk: c_int, hv: c_int, dk: c_int, dv: c_int, act: mlx.mlx_dtype, state: mlx.mlx_dtype, a_log_h: Array, dt_f32: Array, lower_bound: f32) bool,
    run: *const fn (s: mlx.mlx_stream, in: RecurIn) anyerror!RecurOut,
};

/// The gate's operands, borrowed: `a` (the gate's input), `dt_bias`, `A_log` expanded per key channel, its lower bound.
pub const GateIn = struct { a: Array, dt_bias: Array, A_log: Array, lower_bound: f32 };

/// The capture recurrence's operands, borrowed: the scaled heads, the gate, beta, the state before the first row and
/// the row count as a 0-dim int, with the kernels' dtypes and geometry.
pub const CaptureIn = struct { q: Array, k: Array, v: Array, g: Array, beta: Array, state: Array, rows_scalar: Array, in_dtype: mlx.mlx_dtype, out_dtype: mlx.mlx_dtype, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, rows: c_int };
/// New references the trunk owns: y `[B, S, hv, dv]`, the state after the last row, every row's state
/// `[S, B, hv, dv, dk]`.
pub const CaptureOut = struct { y: Array, state: Array, state_seq: Array };

/// A row's projections, given in place of the layer's own.
const Pre = struct { qkv: Array, z: Array, a: Array, b: Array };

/// A serial-rows verify (`Model.serial_rows`): the projections over every row (row-exact through `Model.mm`), then each
/// row's conv, gates, recurrence and gated norm on the one-row decode path from the row before's state, each row's
/// state and the window's conv inputs kept as the capture path keeps them for rollback; then the out projection.
fn serialRows(m: *Model, x: Array, la: *const Kda, e: *Entry, seq_len: c_int) !Array {
    const s = m.s;
    var all: [4]Array = @splat(.{ .ctx = null });
    defer for (all) |t| free(t);
    all[0] = try qkvProj(m, la, x);
    for (all[1..], [_]*const trunk.Linear{ &la.z, &la.a, &la.b }) |*o, l| o.* = try m.mm(l, x);
    var conv_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_in);
    {
        const parts = [_]Array{ e.conv_state, all[0] };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_in, vec, 1, s));
    }
    const ys = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ys);
    const states = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(states);
    var r: c_int = 0;
    while (r < seq_len) : (r += 1) {
        var row: [4]Array = @splat(.{ .ctx = null });
        defer for (row) |t| free(t);
        for (&row, all) |*o, t| o.* = try rowOf(t, r, s);
        const y = try forwardWith(m, x, la, e, 1, 1, false, false, .{ .qkv = row[0], .z = row[1], .a = row[2], .b = row[3] });
        defer _ = mlx.mlx_array_free(y);
        _ = mlx.mlx_vector_array_append_value(ys, y);
        _ = mlx.mlx_vector_array_append_value(states, e.ssm_state);
    }
    free(e.spec_state_seq);
    e.spec_state_seq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_stack_axis(&e.spec_state_seq, states, 0, s));
    free(e.spec_conv_input);
    e.spec_conv_input = conv_in;
    conv_in = mlx.mlx_array_new();
    var y_all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_all);
    try mlx.check(mlx.mlx_concatenate_axis(&y_all, ys, 1, s));
    return m.mm(&la.out, y_all);
}

/// Row `r` of `a` `[1, S, W]`: `[1, 1, W]`.
fn rowOf(a: Array, r: c_int, s: mlx.mlx_stream) !Array {
    const w = mlx.getShape(a)[2];
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &[_]c_int{ 0, r, 0 }, 3, &[_]c_int{ 1, r + 1, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

/// A new reference on `a`.
fn ref(a: Array) Array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, a);
    return r;
}

/// The out projection of `y` (consumed), or with `given` projections (a serial-rows verify's row) `y` itself.
fn outProj(m: *Model, la: *const Kda, y: Array, given: bool) !Array {
    if (given) return y;
    defer _ = mlx.mlx_array_free(y);
    return m.mm(&la.out, y);
}

/// `forward`'s body; `given`: a row's projections, its out projection left to the caller.
fn forwardWith(m: *Model, x: Array, la: *const Kda, e: *Entry, batch: c_int, seq_len: c_int, is_prefill: bool, capture: bool, given: ?Pre) !Array {
    const s = m.s;
    const c = m.c;
    const num_k_heads: c_int = @intCast(c.linear_num_key_heads);
    const num_v_heads: c_int = @intCast(c.linear_num_value_heads);
    const dk: c_int = @intCast(c.linear_key_head_dim);
    const dv: c_int = @intCast(c.linear_value_head_dim);
    const key_dim: c_int = dk * num_k_heads;
    const value_dim: c_int = dv * num_v_heads;
    const conv_dim: c_int = key_dim * 2 + value_dim;
    const kernel: c_int = @intCast(c.linear_conv_kernel_dim);
    const bounded = c.kda_gate_lower_bound != 0.0;

    // Null until assigned: an early error return frees what the defers below see.
    var qkv: Array = .{ .ctx = null };
    var z_proj: Array = .{ .ctx = null };
    // [qkv | z | a | b] from one joined matmul: the fused step reads its columns in place.
    var in_all: Array = .{ .ctx = null };
    defer free(in_all);
    var a_proj: Array = .{ .ctx = null };
    var b_proj: Array = .{ .ctx = null };
    defer _ = mlx.mlx_array_free(qkv);
    defer _ = mlx.mlx_array_free(z_proj);
    defer _ = mlx.mlx_array_free(a_proj);
    defer _ = mlx.mlx_array_free(b_proj);
    if (given) |p| {
        qkv = ref(p.qkv);
        z_proj = ref(p.z);
        a_proj = ref(p.a);
        b_proj = ref(p.b);
    } else if (la.in.built() and (batch * seq_len <= 1 or (batch == 1 and seq_len <= kda_recurrence.MAX_WINDOW))) {
        const all = try m.mm(&la.in.lin, x);
        in_all = all;
        qkv = mlx.mlx_array_new();
        z_proj = mlx.mlx_array_new();
        a_proj = mlx.mlx_array_new();
        b_proj = mlx.mlx_array_new();
        const w = la.in.widths;
        try sliceQkvPart(&qkv, all, 0, w[0], s);
        try sliceQkvPart(&z_proj, all, w[0], w[1], s);
        try sliceQkvPart(&a_proj, all, w[0] + w[1], w[2], s);
        try sliceQkvPart(&b_proj, all, w[0] + w[1] + w[2], w[3], s);
    } else {
        qkv = try qkvProj(m, la, x);
        z_proj = try m.mm(&la.z, x);
        a_proj = try m.mm(&la.a, x);
        b_proj = try m.mm(&la.b, x);
    }
    // KDA decode: one dispatch per layer over the row-joined projection; a verify steps row by row, keeping each
    // row's state (`spec_state_seq`, `spec_conv_input`: the layout rollback reads).
    if (in_all.ctx != null and batch == 1 and seq_len <= kda_recurrence.MAX_WINDOW and la.a_log_h.ctx != null and
        e.initialized and e.ssm_state.ctx != null and e.conv_state.ctx != null)
    step: {
        const w = la.in.widths;
        const r = (try kda_recurrence.decodeStep(s, in_all, w[0], w[0] + w[1], w[0] + w[1] + w[2], num_v_heads, e.conv_state, la.conv_w, la.a_log_h, la.dt_f32, e.ssm_state, la.norm_w, low(m, &la.a2), low(m, &la.z2), c.kda_gate_lower_bound, c.rms_norm_eps, capture)) orelse break :step;
        defer _ = mlx.mlx_array_free(r.y);
        if (capture) {
            // The window's conv inputs behind the 3 that preceded it, and the state after every row.
            var qkv_rows = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qkv_rows);
            try mlx.check(mlx.mlx_slice(&qkv_rows, in_all, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, seq_len, w[0] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const parts = [_]Array{ e.conv_state, qkv_rows };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            free(e.spec_conv_input);
            e.spec_conv_input = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&e.spec_conv_input, vec, 1, s));
            free(e.spec_state_seq);
            e.spec_state_seq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&e.spec_state_seq, r.state_seq, &[_]c_int{ seq_len, 1, num_v_heads, dv, dk }, 5, s));
            _ = mlx.mlx_array_free(r.state_seq);
        }
        _ = mlx.mlx_array_free(e.conv_state);
        e.conv_state = r.conv_state;
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = r.state;
        return m.mm(&la.out, r.y);
    }
    // Low-rank gates: the projections above are the first stage; the joined buffer's a/z columns are then
    // first-stage values, so readers lose it.
    if (la.a2.w.ctx != null or la.z2.w.ctx != null) {
        if (la.a2.w.ctx != null) {
            const full = try m.mm(&la.a2, a_proj);
            _ = mlx.mlx_array_free(a_proj);
            a_proj = full;
        }
        if (la.z2.w.ctx != null) {
            const full = try m.mm(&la.z2, z_proj);
            _ = mlx.mlx_array_free(z_proj);
            z_proj = full;
        }
        free(in_all);
        in_all = .{ .ctx = null };
    }
    if (m.kda.q_scale.ctx == null) {
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(c.linear_key_head_dim));
        m.kda.q_scale = try scalarOf(inv, mlx.mlx_array_dtype(qkv), s);
        m.kda.k_scale = try scalarOf(@sqrt(inv), mlx.mlx_array_dtype(qkv), s);
    }
    const inv_scale_sq = m.kda.q_scale;
    const inv_sqrt_sc = m.kda.k_scale;

    var q_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled);
    var k_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_scaled);
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    var fused_prework = false;
    var beta_fused: ?Array = null;

    // The per-core recurrence builds the per-channel gate itself, so the prework skips its per-head one.
    const kda_fused = bounded and !capture and
        kda_recurrence.serves(s, num_k_heads, num_v_heads, dk, dv, mlx.mlx_array_dtype(qkv), m.ssmStateDtype());
    // Wide prework is token-local; a cold chunk supplies the same zero history as the composed conv.
    const prefill_fused = kda_fused and is_prefill and decode_fused.gdnPrefillFusedFor(seq_len, batch);
    const stages = m.kda.stages;
    const prompt_rows = bounded and is_prefill and !capture and batch == 1 and seq_len >= 17 and !kda_fused;
    const verify_rows = bounded and capture and batch == 1;
    const prologue: ?PrologueOut = if (prompt_rows and kernel == 4) if (stages.prologue) |f|
        try f(s, .{ .qkv = qkv, .conv_state = if (e.initialized) e.conv_state else null, .conv_w = la.conv_w, .q_scale = inv_scale_sq, .k_scale = inv_sqrt_sc, .qk_eps = qkNormEps(m), .hk = num_k_heads, .dk = dk, .hv = num_v_heads, .dv = dv, .taps = kernel })
    else
        null else null;
    if (prologue) |p| {
        free(e.conv_state);
        e.conv_state = p.window;
        e.initialized = true;
        _ = mlx.mlx_array_free(q_scaled);
        _ = mlx.mlx_array_free(k_scaled);
        _ = mlx.mlx_array_free(v_heads);
        q_scaled = p.q;
        k_scaled = p.k;
        v_heads = p.v;
    }
    // A model's own verify prework: beta rides `beta_fused`; the stock fused epilogue stays the fused prework's.
    const vprework: ?PreworkOut = if (verify_rows and kernel == 4) if (stages.verify_prework) |f|
        try f(s, .{ .qkv = qkv, .b = b_proj, .conv_state = if (e.initialized) e.conv_state else null, .conv_w = la.conv_w, .q_scale = inv_scale_sq, .k_scale = inv_sqrt_sc, .qk_eps = qkNormEps(m), .hk = num_k_heads, .dk = dk, .hv = num_v_heads, .dv = dv, .taps = kernel, .rows = seq_len })
    else
        null else null;
    if (vprework) |p| {
        free(e.conv_state);
        e.conv_state = p.window;
        e.initialized = true;
        free(e.spec_conv_input);
        e.spec_conv_input = p.conv_input;
        _ = mlx.mlx_array_free(q_scaled);
        _ = mlx.mlx_array_free(k_scaled);
        _ = mlx.mlx_array_free(v_heads);
        q_scaled = p.q;
        k_scaled = p.k;
        v_heads = p.v;
        beta_fused = p.beta;
    }
    if (kda_fused and (batch * seq_len <= decode_fused.GDN_FUSED_MAX_ROWS or prefill_fused) and
        kernel == 4 and (e.initialized or prefill_fused))
    blk: {
        var zero_conv: Array = .{ .ctx = null };
        defer free(zero_conv);
        const cold = !e.initialized;
        if (cold) {
            zero_conv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero_conv, &[_]c_int{ batch, 3, conv_dim }, 3, mlx.mlx_array_dtype(qkv), s));
        }
        const incoming_conv = if (cold) zero_conv else e.conv_state;
        const pre = (decode_fused.gdnPreworkFused(s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = b_proj,
            .b_off = 0,
            .b_stride = num_v_heads,
            .a = a_proj,
            .a_off = 0,
            .a_stride = num_v_heads,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .conv_state = incoming_conv,
            .conv_w = la.conv_w,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .qk_eps = qkNormEps(m),
            .hk = num_k_heads,
            .hv = num_v_heads,
            .dk = dk,
            .dv = dv,
            .seq = seq_len,
            .batch = batch,
            .gate = false,
            .g_dtype = gdn_decode.gateDtype(mlx.mlx_array_dtype(qkv), m.ssmStateDtype()),
        }) catch null) orelse break :blk;
        free(e.conv_state);
        e.conv_state = pre.conv_state;
        e.initialized = true;
        if (prefill_fused and seq_len >= 512 and !decode_fused.gdn_prefill_engaged) {
            decode_fused.gdn_prefill_engaged = true;
            log.info("[gdn-prefill] engaged: S={d} B={d} cold={}\n", .{ seq_len, batch, cold });
        }
        _ = mlx.mlx_array_free(q_scaled);
        _ = mlx.mlx_array_free(k_scaled);
        _ = mlx.mlx_array_free(v_heads);
        q_scaled = pre.q;
        k_scaled = pre.k;
        v_heads = pre.v;
        fused_prework = true;
        _ = mlx.mlx_array_free(pre.g);
        beta_fused = pre.beta;
    }

    if (!fused_prework and prologue == null and vprework == null) {
        var g = ops.Graph.init(std.heap.c_allocator, s);
        defer g.deinit();
        const conv = try conv1d.run(&g, qkv, if (e.initialized) e.conv_state else null, la.conv_w, null, batch, conv_dim, kernel, true);
        free(e.conv_state);
        e.conv_state = g.escape(conv.window);
        e.initialized = true;
        if (capture) {
            free(e.spec_conv_input);
            e.spec_conv_input = g.escape(conv.input);
        }
        const conv_out = conv.y;
        const q_flat = try g.slice(conv_out, &.{ 0, 0, 0 }, &.{ batch, seq_len, key_dim });
        const k_flat = try g.slice(conv_out, &.{ 0, 0, key_dim }, &.{ batch, seq_len, key_dim * 2 });
        const v_flat = try g.slice(conv_out, &.{ 0, 0, key_dim * 2 }, &.{ batch, seq_len, key_dim * 2 + value_dim });
        const q_heads = try g.reshape(q_flat, &.{ batch, seq_len, num_k_heads, dk });
        const k_heads = try g.reshape(k_flat, &.{ batch, seq_len, num_k_heads, dk });
        _ = mlx.mlx_array_free(v_heads);
        v_heads = g.escape(try g.reshape(v_flat, &.{ batch, seq_len, num_v_heads, dv }));
        // q = (1/dk) * rms_norm(q), k = (1/sqrt(dk)) * rms_norm(k), over a weight of ones.
        if (m.kda.ones.ctx == null) {
            var w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&w, &[_]c_int{dk}, 1, mlx.mlx_array_dtype(qkv), s));
            m.kda.ones = w;
        }
        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, m.kda.ones, qkNormEps(m), s));
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, inv_scale_sq, s));
        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, m.kda.ones, qkNormEps(m), s));
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, inv_sqrt_sc, s));
    }

    // A model's own prompt recurrence where it serves these rows; it makes the gate itself.
    const recur_stage: ?Recurrence = if (prompt_rows and mlx.streamIsGpu(s)) if (stages.recurrence) |r|
        (if (r.serves(s, num_k_heads, num_v_heads, dk, dv, mlx.mlx_array_dtype(qkv), m.ssmStateDtype(), la.a_log_h, la.dt_f32, c.kda_gate_lower_bound)) r else null)
    else
        null else null;
    // One gate value per key channel: the bounded sigmoid where the config bounds it (fla's two arms), else the
    // softplus chain over `A_log` expanded per channel. The per-core recurrence and a model's own one build the gate
    // themselves; a model's own verify gate goes first where it serves.
    const g = if (kda_fused or recur_stage != null) Array{ .ctx = null } else if (bounded) pick: {
        if (verify_rows) if (stages.verify_gate) |f| if (try f(s, .{ .a = a_proj, .dt_bias = la.dt_bias, .A_log = la.A_log, .lower_bound = c.kda_gate_lower_bound })) |vg| break :pick vg;
        break :pick try kdaGateChain(la.A_log, a_proj, la.dt_bias, c.kda_gate_lower_bound, s);
    } else try softplusGate(m, la.A_log, a_proj, la.dt_bias);
    defer free(g);

    const beta = beta_fused orelse blk: {
        var bt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&bt, b_proj, s));
        break :blk bt;
    };
    defer _ = mlx.mlx_array_free(beta);

    if (e.ssm_state.ctx == null) {
        e.ssm_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&e.ssm_state, &[_]c_int{ batch, num_v_heads, dv, dk }, 4, m.ssmStateDtype(), s));
    }

    const T_scalar = mlx.mlx_array_new_int(seq_len);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
    // The kernels specialize on the actual widths: y in the activation dtype, the state in its own.
    const in_dtype = mlx.mlx_array_dtype(q_scaled);
    const state_dtype = mlx.mlx_array_dtype(e.ssm_state);
    const out_dtype: mlx.mlx_dtype = if (in_dtype == .float16) .float16 else .bfloat16;

    const kcfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(kcfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &y_shape, 4, out_dtype));

    var y_bthd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_bthd);
    if (!kda_fused and !mlx.streamIsGpu(s)) {
        const r = (try kernels.gdnRecurCuda(s, q_scaled, k_scaled, v_heads, g, beta, e.ssm_state, true, capture, out_dtype)) orelse
            try kernels.gdnRecurOps(s, q_scaled, k_scaled, v_heads, g, beta, e.ssm_state, true, capture, out_dtype);
        _ = mlx.mlx_array_free(y_bthd);
        y_bthd = r.y;
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = r.state;
        if (capture) {
            free(e.spec_state_seq);
            e.spec_state_seq = r.state_seq;
        }
    } else if (if (verify_rows) if (stages.verify_capture) |f| try f(s, .{ .q = q_scaled, .k = k_scaled, .v = v_heads, .g = g, .beta = beta, .state = e.ssm_state, .rows_scalar = T_scalar, .in_dtype = in_dtype, .out_dtype = out_dtype, .hk = num_k_heads, .hv = num_v_heads, .dk = dk, .dv = dv, .batch = batch, .rows = seq_len }) else null else null) |r| {
        _ = mlx.mlx_array_free(y_bthd);
        y_bthd = r.y;
        free(e.spec_state_seq);
        e.spec_state_seq = r.state_seq;
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = r.state;
    } else if (capture) {
        // Per-row states for rollback: state_seq [T, B, Hv, Dv, Dk] (its last row unwritten), the final state its
        // own output.
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ seq_len, batch, num_v_heads, dv, dk }, 5, state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ batch, num_v_heads, dv, dk }, 4, state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(kcfg, 32, dv, batch * num_v_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(kcfg, 32, 4, 1));
        try templateArgs(kcfg, in_dtype, state_dtype, out_dtype, dk, dv, num_k_heads, num_v_heads);
        const seq_stride = mlx.mlx_array_new_int(batch * num_v_heads * dv * dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const inputs = [_]Array{ q_scaled, k_scaled, v_heads, g, beta, e.ssm_state, T_scalar, seq_stride };
        const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, try kernels.getGdnKernelSeq(true), in_vec, kcfg, s));
        if (mlx.mlx_vector_array_size(out_vec) != 3) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&y_bthd, out_vec, 0));
        var state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&state_seq, out_vec, 1));
        free(e.spec_state_seq);
        e.spec_state_seq = state_seq;
        var final_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&final_state, out_vec, 2));
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = final_state;
    } else if (recur_stage) |rs| {
        const r = try rs.run(s, .{ .q = q_scaled, .k = k_scaled, .v = v_heads, .a = a_proj, .beta = beta, .a_log_h = la.a_log_h, .dt_f32 = la.dt_f32, .lower_bound = c.kda_gate_lower_bound, .state = e.ssm_state });
        _ = mlx.mlx_array_free(y_bthd);
        y_bthd = r.y;
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = r.state;
    } else if (kda_fused) {
        const r = try kda_recurrence.recur(s, q_scaled, k_scaled, v_heads, a_proj, beta, la.a_log_h, la.dt_f32, c.kda_gate_lower_bound, e.ssm_state);
        _ = mlx.mlx_array_free(y_bthd);
        y_bthd = r.y;
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = r.state;
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(kcfg, &[_]c_int{ batch, num_v_heads, dv, dk }, 4, state_dtype));
        // A per-channel gate is another indexing contract than the blocked kernel's per-head one: stock kernel.
        try kernels.gdnSetGrid(kcfg, .stock, dv, num_v_heads, batch);
        try templateArgs(kcfg, in_dtype, state_dtype, out_dtype, dk, dv, num_k_heads, num_v_heads);
        const inputs = [_]Array{ q_scaled, k_scaled, v_heads, g, beta, e.ssm_state, T_scalar };
        const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, try kernels.gdnRouteKernel(.stock, 0, true), in_vec, kcfg, s));
        if (mlx.mlx_vector_array_size(out_vec) != 2) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&y_bthd, out_vec, 0));
        var new_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&new_state, out_vec, 1));
        _ = mlx.mlx_array_free(e.ssm_state);
        e.ssm_state = new_state;
    }

    // rms_norm(y) * sigmoid(z) straight to the out projection's input (decode and verify widths).
    if (fused_prework) {
        if (m.kda.eps.ctx == null) m.kda.eps = mlx.mlx_array_new_float(c.rms_norm_eps);
        if (try decode_fused.gdnNormGateFused(s, y_bthd, z_proj, 0, value_dim, la.norm_w, m.kda.eps, false, num_v_heads, dv, batch, seq_len)) |flat|
            return outProj(m, la, flat, given != null);
    }
    // A model's own epilogue (`!fused_prework` holds on these rows already: kept to say so).
    if (!fused_prework and (prompt_rows or verify_rows)) if (stages.epilogue) |f| {
        if (m.kda.eps.ctx == null) m.kda.eps = mlx.mlx_array_new_float(c.rms_norm_eps);
        if (try f(s, .{ .y = y_bthd, .z = z_proj, .norm_w = la.norm_w, .eps = m.kda.eps, .hv = num_v_heads, .dv = dv, .batch = batch, .rows = seq_len, .verify = verify_rows })) |flat|
            return outProj(m, la, flat, given != null);
    };
    var z_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &y_shape, 4, s));
    // KDA's gated norm takes a plain sigmoid of the output gate (fla's FusedRMSNormGated, `y * sigmoid(z)`).
    const y_normed = try m.rmsNorm(y_bthd, la.norm_w);
    defer _ = mlx.mlx_array_free(y_normed);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, s));
    var gated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gated);
    try mlx.check(mlx.mlx_multiply(&gated, y_normed, gate, s));
    var out_flat = mlx.mlx_array_new();
    mlx.check(mlx.mlx_reshape(&out_flat, gated, &[_]c_int{ batch, seq_len, value_dim }, 3, s)) catch |err| {
        _ = mlx.mlx_array_free(out_flat);
        return err;
    };
    return outProj(m, la, out_flat, given != null);
}

/// The KDA capture and route kernels' template arguments (a model's own capture kernel shares them).
pub fn templateArgs(kcfg: mlx.mlx_fast_metal_kernel_config, in_dt: mlx.mlx_dtype, st_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, dk: c_int, dv: c_int, hk: c_int, hv: c_int) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "InT", in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "StT", st_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(kcfg, "OutT", out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, "Dk", dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, "Dv", dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, "Hk", hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(kcfg, "Hv", hv));
}

/// q|k|v through the joined projection, or each part with the three joined along the last axis.
fn qkvProj(m: *const Model, la: *const Kda, x: Array) !Array {
    if (la.qkv.w.ctx != null) return m.mm(&la.qkv, x);
    var parts: [3]Array = @splat(.{ .ctx = null });
    defer for (parts) |p| free(p);
    for (&parts, &la.qkv_parts) |*out, *q| out.* = try m.mm(q, x);
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var qkv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qkv);
    try mlx.check(mlx.mlx_concatenate_axis(&qkv, vec, -1, m.s));
    return qkv;
}
