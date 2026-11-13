//! The KDA hybrids' loaded trunk (GLM-5.3-Flash, Ling 3.0) and the helpers their forwards share. Read from the
//! engine's resident source in the legacy load's layout: KDA layers with q|k|v and their convs joined (GLM's decode
//! row group too), sparse attention (GLM) or naive MLA (Ling), the dense first layers, the routed MoE as the shared
//! block reads it, each GLM sublayer's hyper-connection, GLM's `mtp.0` head. Dense projections are pre-transposed;
//! quant params resolve per weight at its input width, as the engine's resolver does on first touch.
//! copied_from: engines/mlx/load/layers_layered.zig (`initMoeLayers`, its GLM and bailing arms),
//! engines/mlx/load/weights.zig (`loadRoutedMoeAt`, `f32Table`), engines/mlx/load/init.zig (the embedding and the
//! head), sdk/quant/params.zig (`computeQuantParams`), plugins/glm5_next/load.zig, plugins/glm5_next/mtp.zig
//! (`Head.load`), sdk/blocks/embedding.zig (`rawEmbedding`), sdk/blocks/linear/qmatmul.zig (`lmHeadProject`),
//! sdk/blocks/mlp/dense.zig (`denseMLP`), plugins/glm5_next/forward.zig (`glmHeadMm`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const array = @import("../../mlx/array.zig");
const log = @import("../../sdk/util/log.zig");
const linear = @import("../../sdk/bases/blocks/linear.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const streamed = @import("../../sdk/weights/streamed.zig");
const outside = @import("../../sdk/quant/outside.zig");
const outside_model = @import("../../sdk/quant/outside_model.zig");
const routed_mod = @import("../../sdk/weights/routed.zig");
const narrowsLoadedF16 = @import("../../sdk/weights/load.zig").narrowsLoadedF16;
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const gdn_base = @import("../../sdk/bases/blocks/gdn.zig");
const exl3 = @import("../../sdk/quant/exl3/format.zig");
const params = @import("../../sdk/quant/params.zig");
const fp8_block128 = @import("../../sdk/quant/fp8_block128.zig");
const fused_rows = @import("../../sdk/blocks/linear/fused_rows.zig");
const qmatmulBits = @import("../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const clampedSwiGLU = @import("../../sdk/blocks/mlp/activations.zig").clampedSwiGLU;
const add_norm = @import("../../sdk/blocks/norms/add_norm.zig");
const residual_norm = @import("../../sdk/blocks/norms/residual_norm.zig");
const config = @import("config.zig");
const glm5 = @import("glm5_next.zig");

const Array = mlx.mlx_array;
pub const Linear = linear.Linear;
const QuantMode = @FieldType(Linear, "mode");
const QuantParams = params.QuantParams;
const Config = config.Config;
const gpa = std.heap.c_allocator;

/// One DeepSeek-V4 Sinkhorn hyper-connection (`attn_hc` / `ffn_hc`), f32 tables.
pub const Mhc = struct {
    /// `[(2+hc)*hc, hc*hidden]`
    fn_rows: Array,
    /// `[(2+hc)*hc]`
    base: Array,
    /// `[3]`: pre / post / comb.
    scale: Array,
};

/// DeepSeek sparse attention: absorbed NoPE MLA over a latent, plus the pooled indexer that picks the tokens each
/// query reads.
pub const Dsa = struct {
    q_a: Linear,
    kv_a: Linear,
    q_a_norm: Array,
    kv_a_norm: Array,
    q_b: Linear,
    /// `[H, latent, head_dim]` (dense kept as stored): folds a query into the latent, or lifts the latent to keys.
    embed_q: Linear,
    /// `[H, head_dim, latent]`: lifts the attended latent to values.
    unembed: Linear,
    o: Linear,
    idx_wq_b: Linear,
    idx_wk: Linear,
    idx_k_norm_w: Array,
    idx_k_norm_b: Array,
    idx_wproj: Linear,
    /// f32 `[hidden, index_head_dim]` (the pool gate projection, transposed).
    idx_gate_t: Array,
    /// f32 `[kpool, index_head_dim]` (the per-slot gate offset inside a pool).
    idx_ape: Array,
};

/// A KDA layer: q|k|v (and their depthwise convs) joined, the per-channel gate `a`, beta `b`, the output gate `z`.
pub const Kda = struct {
    /// q|k|v as one projection; null `w` when they ship at different widths (`qkv_parts`).
    qkv: Linear = .{},
    qkv_parts: [3]Linear = @splat(.{}),
    z: Linear,
    a: Linear,
    b: Linear,
    out: Linear,
    /// GLM's low-rank second stages: `a = a2(a(x))`, `z = z2(z(x))`; null `w` = full rank (Ling).
    a2: Linear = .{},
    z2: Linear = .{},
    /// `[qkv | z | a | b]` rows joined (GLM, quantized); the parts are views of it.
    in: linear.Joined = .{},
    conv_w: Array,
    /// `A_log` expanded per key channel `[Hv*Dk]`.
    A_log: Array,
    dt_bias: Array,
    norm_w: Array,
    /// GLM's f32 copies the fused KDA kernels read: `A_log` per head `[Hv]`, `dt_bias` `[Hv*Dk]`.
    a_log_h: Array = .{},
    dt_f32: Array = .{},
};

/// Ling's naive MLA: the low-rank query pair (tiny) or a direct `q_proj` (flash), the compressed KV latent with its
/// shared rope key, the head-wise output gate.
pub const Mla = struct {
    q: Linear = .{},
    q_a: Linear = .{},
    q_a_norm: Array = .{},
    q_b: Linear = .{},
    kv_a: Linear,
    kv_a_norm: Array,
    kv_b: Linear,
    g: Linear,
    o: Linear,
};

/// A dense SwiGLU layer; gate|up joined rows read at one row.
pub const Dense = struct { gate: Linear, up: Linear, down: Linear, gu: linear.Joined = .{} };

pub const Layer = struct {
    input_norm: Array,
    post_norm: Array,
    attn_hc: ?Mhc = null,
    ffn_hc: ?Mhc = null,
    mixer: union(enum) { kda: Kda, dsa: Dsa, mla: Mla },
    mlp: union(enum) { dense: Dense, moe: moe.Moe.Weights },
};

/// GLM's `mtp.0` head: `eh_proj(cat[enorm(embed(x)), hnorm(h)])` through one plain DSA + MoE block, then `norm`.
pub const Head = struct {
    enorm: Array,
    hnorm: Array,
    norm: Array,
    input_norm: Array,
    post_norm: Array,
    eh_proj: Linear,
    dsa: Dsa,
    moe: moe.Moe.Weights,
};

/// Ling-3.1's NextN (MTP) layer, `<prefix>.layers.<num_hidden_layers>` (bailing_hybrid/head.zig):
/// `eh_proj(cat[enorm(embed(x)), hnorm(h)])` through one MLA + MoE block, then `final_layernorm`. A streaming pack's
/// layer is resident: its experts are the claiming plugin's format banks (`Load.experts`), not the bank's.
pub const LingHead = struct {
    enorm: Array,
    hnorm: Array,
    final_norm: Array,
    input_norm: Array,
    post_norm: Array,
    eh_proj: Linear,
    mla: Mla,
    moe: moe.Moe.Weights,
};

/// The KDA layers' constants, built on first use in the projections' dtype (the legacy `gdn_*` caches).
pub const KdaConsts = struct {
    q_scale: Array = .{},
    k_scale: Array = .{},
    ones: Array = .{},
    eps: Array = .{},
    /// The softplus gate chain compiled (`gdn.compileGate`), for a gate without a lower bound (Ling).
    gate_fn: compiled.Closure = .{ .ctx = null },
};

pub const Model = struct {
    c: *const Config,
    s: mlx.mlx_stream,
    caps: abi.Caps,
    embed: Linear,
    head: Linear,
    final_norm: Array,
    layers: []Layer = &.{},
    /// The `mtp.0` head's weights, read at load; served once `HeadHooks.load` evaluated them.
    mtp: ?Head = null,
    mtp_ready: bool = false,
    /// Ling's NextN layer (`LingHead`); served once `readyLingHead` evaluated it.
    ling_mtp: ?LingHead = null,
    /// The head drafts read (`host.trunkWeight(.head)`) where the trunk's runs on the outside plugin (`Linear.ext`,
    /// which the engine cannot read): its rows as MLX affine 4-bit (`draftReadout`). The trunk's logits keep the
    /// plugin's head, so a verify's text is unchanged.
    draft_head: Linear = .{},
    kda: KdaConsts = .{},
    /// `rms_norm_eps` as an array (the add+norm kernel's).
    rms_eps: Array = .{},
    owned: std.ArrayList(Array) = .empty,
    /// The trunk's routed experts streamed from the pack (the `stream_experts` setting); the head's stay resident.
    experts: ?streamed.Streamed = null,
    /// The weights an outside quant plugin loaded (a pack it claims), held until unload.
    quant: outside_model.Load = .{ .a = gpa },
    /// A streaming pack's bank (`Config.streamed_bank`), bound to every routed layer; the head's experts stay resident.
    bank: ?*outside.Bank = null,
    /// The bank as the routed layers read it.
    routed: ?routed_mod.Routed = null,
    /// Set for one forward whose rows must each carry a one-row decode step's bytes (bailing_hybrid's exact verify):
    /// `mm` runs rows in calls of their projection's row-exact width, KDA, MLA and the MoE row by row.
    serial_rows: bool = false,

    pub fn deinit(m: *Model) void {
        if (m.experts) |*x| x.close();
        if (m.bank) |b| b.close();
        m.quant.deinit();
        for (m.layers) |*l| if (l.mlp == .moe) compiled.free(&l.mlp.moe.route_fn);
        if (m.mtp) |*h| compiled.free(&h.moe.route_fn);
        if (m.ling_mtp) |*h| compiled.free(&h.moe.route_fn);
        inline for (.{ m.kda.q_scale, m.kda.k_scale, m.kda.ones, m.kda.eps, m.rms_eps }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        compiled.free(&m.kda.gate_fn);
        for (m.owned.items) |a| _ = mlx.mlx_array_free(a);
        m.owned.deinit(gpa);
        gpa.free(m.layers);
    }

    pub fn ssmStateDtype(m: *const Model) mlx.mlx_dtype {
        return if (m.c.isGlm5()) .float32 else .bfloat16;
    }

    /// `x @ l^T` at `l`'s own quant geometry; under `serial_rows` in calls of `l`'s row-exact width.
    pub fn mm(m: *const Model, l: *const Linear, x: Array) !Array {
        if (m.serial_rows) return linear.eachRows(x, l.exactRows(), m.s, MmCall{ .m = m, .l = l }, MmCall.run);
        return m.mmAll(l, x);
    }

    const MmCall = struct {
        m: *const Model,
        l: *const Linear,
        fn run(c: MmCall, x: Array) anyerror!Array {
            return c.m.mmAll(c.l, x);
        }
    };

    fn mmAll(m: *const Model, l: *const Linear, x: Array) !Array {
        if (l.ext) |*e| return e.matmul(x, m.s);
        return qmatmulBits(x, l.w, l.s, l.b, l.bits, l.group_size, l.mode, m.s);
    }

    pub fn rmsNorm(m: *const Model, x: Array, w: Array) !Array {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, m.c.rms_norm_eps, m.s));
        return out;
    }

    /// `h + delta` and its RMSNorm in one dispatch at decode and verify widths; null: the caller keeps the ops.
    pub fn narrowAddNorm(m: *const Model, h: Array, delta: Array, w: Array) !?add_norm.Out {
        if (w.ctx == null) return null;
        const sh = mlx.getShape(h);
        if (sh.len != 3 or sh[0] * sh[1] > residual_norm.add_norm_max_rows) return null;
        return add_norm.addNorm(h, delta, w, m.rms_eps, m.s);
    }

    /// The embedding rows of `ids` `[B, L]`: `[B, L, hidden]` bf16.
    pub fn embedding(m: *const Model, ids: Array) !Array {
        const s = m.s;
        const e = &m.embed;
        const sh = mlx.getShape(ids);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, ids, &[_]c_int{sh[0] * sh[1]}, 1, s));
        var tw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tw);
        try mlx.check(mlx.mlx_take_axis(&tw, e.w, flat, 0, s));
        var emb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb);
        if (e.s.ctx == null) {
            try mlx.check(mlx.mlx_astype(&emb, tw, .bfloat16, s));
        } else {
            var ts = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ts);
            try mlx.check(mlx.mlx_take_axis(&ts, e.s, flat, 0, s));
            var tb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tb);
            if (e.b.ctx != null) try mlx.check(mlx.mlx_take_axis(&tb, e.b, flat, 0, s));
            try mlx.check(mlx.mlx_dequantize(&emb, tw, ts, tb, mlx.mlx_optional_int.some(@intCast(e.group_size)), mlx.mlx_optional_int.some(@intCast(e.bits)), e.mode.cstr(), .{}, .{ .value = .bfloat16, .has_value = true }, s));
        }
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, emb, &[_]c_int{ sh[0], sh[1], @intCast(m.c.hidden_size) }, 3, s));
        return out;
    }

    /// The logits of `x`: a dense head `[vocab, hidden]` is read through a transposed view (the embedding it may
    /// alias stays `[vocab, hidden]`).
    pub fn lmHead(m: *const Model, x: Array) !Array {
        if (m.head.s.ctx != null or m.head.ext != null) return m.mm(&m.head, x);
        const wt = try transposeLast(m.head.w, m.s);
        defer _ = mlx.mlx_array_free(wt);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_matmul(&out, x, wt, m.s));
        return out;
    }

    /// The dense SwiGLU (clamped at `swiglu_limit` where the config clamps); one row reads gate|up joined.
    pub fn denseMlp(m: *const Model, dw: *const Dense, x: Array) !Array {
        const s = m.s;
        const xsh = mlx.getShape(x);
        var rows: c_int = 1;
        for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        // Serial rows take the one-row arm: wider rows clamp where it does not.
        if (dw.gu.built() and (rows <= fused_rows.FUSED_ROWS_MAX_M or m.serial_rows)) {
            const both = try m.mm(&dw.gu.lin, x);
            defer _ = mlx.mlx_array_free(both);
            if (dw.gu.widths[0] == dw.gu.widths[1]) {
                // The legacy joined arm: unclamped even where the config clamps.
                if (try fused_act.swigluJoined(s, both)) |act| {
                    defer _ = mlx.mlx_array_free(act);
                    return m.mm(&dw.down, act);
                }
            }
            try fused_rows.sliceQkvPart(&gate, both, 0, dw.gu.widths[0], s);
            try fused_rows.sliceQkvPart(&up, both, dw.gu.widths[0], dw.gu.widths[1], s);
        } else {
            gate = try m.mm(&dw.gate, x);
            up = try m.mm(&dw.up, x);
        }
        const act = try m.swiglu(gate, up);
        defer _ = mlx.mlx_array_free(act);
        return m.mm(&dw.down, act);
    }

    fn swiglu(m: *const Model, gate: Array, up: Array) !Array {
        const s = m.s;
        if (m.c.swiglu_clamp) return (try fused_act.fusedClampedSwiGLU(s, gate, up, m.c.swiglu_limit)) orelse
            clampedSwiGLU(s, gate, up, m.c.swiglu_limit);
        if (try fused_act.fusedSwiGLU(s, gate, up)) |y| return y;
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_multiply(&act, gate, sig, s));
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_multiply(&out, act, up, s));
        return out;
    }

    /// Per-head projection over `q` `[H, out, in]`: `x @ w^T` when `transpose`, else `x @ w`.
    pub fn headMm(m: *const Model, x: Array, q: *const Linear, transpose: bool) !Array {
        const s = m.s;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        if (q.s.ctx == null) {
            if (transpose) {
                var wt = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wt);
                try mlx.check(mlx.mlx_swapaxes(&wt, q.w, -1, -2, s));
                try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
            } else try mlx.check(mlx.mlx_matmul(&out, x, q.w, s));
            return out;
        }
        if (transpose and q.mode == .affine) if (try glm5.headQmv(s, x, q.w, q.s, q.b, q.bits, q.group_size, 1.0)) |y| {
            _ = mlx.mlx_array_free(out);
            return y;
        };
        try mlx.check(mlx.mlx_quantized_matmul(&out, x, q.w, q.s, q.b, transpose, mlx.mlx_optional_int.some(@intCast(q.group_size)), mlx.mlx_optional_int.some(@intCast(q.bits)), q.mode.cstr(), s));
        return out;
    }
};

/// `w` with its last two axes swapped (a view).
fn transposeLast(w: Array, s: mlx.mlx_stream) !Array {
    const nd = mlx.getShape(w).len;
    var perm: [8]c_int = undefined;
    for (0..nd) |i| perm[i] = @intCast(i);
    perm[nd - 2] = @intCast(nd - 1);
    perm[nd - 1] = @intCast(nd - 2);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, w, &perm, @intCast(nd), s));
    return out;
}

// ── quant params ──

fn modeOf(m: config.QuantMode) QuantMode {
    return std.meta.stringToEnum(QuantMode, @tagName(m)).?;
}

/// A quantized weight's (bits, group_size, mode) at input width `in_dim` (null: unknown), from its packed geometry
/// and the pack's declaration; a dense weight's are the declaration's.
pub fn resolve(c: *const Config, w: Array, sc: Array, in_dim: ?u32) QuantParams {
    const declared: QuantParams = .{ .bits = c.quant_bits, .group_size = c.quant_group_size, .mode = modeOf(c.quant_mode) };
    if (sc.ctx == null) return declared;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;
    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (c.quant_mode != .affine) {
            if (in_dim) |in| {
                if (fpFromGeometry(c, w_cols, s_cols, in)) |qp| return qp;
            } else if (s_cols > 0 and (w_cols * 32) % s_cols == 0) {
                const per_group = (w_cols * 32) / s_cols;
                const bits: u32 = if (per_group == 256) 8 else 4;
                const gs = per_group / bits;
                if (fpModeFor(bits, gs)) |mode| return .{ .bits = bits, .group_size = gs, .mode = mode };
            }
            return declared;
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }
    if (c.quant_mode == .affine) {
        if (in_dim) |in| if (params.affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        const bits: u32 = if (w_shape.len < 2 or s_shape.len < 2 or s_cols == 0) 4 else (w_cols * 32) / (s_cols * c.quant_group_size);
        return .{ .bits = bits, .group_size = c.quant_group_size, .mode = .affine };
    }
    if (in_dim) |in| if (in > 0 and s_cols > 0 and w_cols > 0) return .{ .bits = (w_cols * 32) / in, .group_size = in / s_cols, .mode = .affine };
    if (w_cols > 0 and s_cols > 0) return .{ .bits = @max((w_cols * 32) / s_cols / 64, 1), .group_size = 64, .mode = .affine };
    return .{ .bits = 8, .group_size = 64, .mode = .affine };
}

fn fpModeFor(bits: u32, group_size: u32) ?QuantMode {
    if (bits == 8 and group_size == 32) return .mxfp8;
    if (bits == 4 and group_size == 32) return .mxfp4;
    if (bits == 4 and group_size == 16) return .nvfp4;
    return null;
}

/// An fp tensor's own (bits, group_size, mode) where its geometry disagrees with the declaration.
fn fpFromGeometry(c: *const Config, w_cols: u32, s_cols: u32, in_dim: u32) ?QuantParams {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = w_cols * 32;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const group_size = in_dim / s_cols;
    if (bits == c.quant_bits and group_size == c.quant_group_size) return null;
    const mode = fpModeFor(bits, group_size) orelse return null;
    return .{ .bits = bits, .group_size = group_size, .mode = mode };
}

// ── load ──

/// A weight triple as stored (`<base>.{weight,scales,biases}`), scales and biases absent on a dense weight; or a
/// projection the pack's outside quant plugin loaded (`ext`, `w` its first array).
const Triple = struct { w: Array, s: Array = .{}, b: Array = .{}, ext: ?outside.Weight = null };

const Loader = struct {
    src: *const quant_abi.Source,
    c: *const Config,
    s: mlx.mlx_stream,
    prefix: []const u8,
    owned: *std.ArrayList(Array),
    exl3: ?exl3.Spec,
    experts: ?*streamed.Streamed = null,
    /// The outside quant plugin claiming the pack (`format` null: none).
    quant: *outside_model.Load,

    /// Takes `a`, an array this load made.
    fn own(l: *Loader, a: Array) !Array {
        if (a.ctx != null) try l.owned.append(gpa, a);
        return a;
    }

    /// A reference of the model's own on `a` (a source tensor, or one this load made).
    fn keep(l: *Loader, a: Array) !Array {
        if (a.ctx == null) return a;
        var h = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_array_set(&h, a));
        return l.own(h);
    }

    /// The source's tensor `name`, borrowed for the load; empty when the pack has none. A pack an outside plugin
    /// claims loads as stored: its f16 tables and scales narrow here as the engine narrows every other pack's.
    fn get(l: *Loader, name: []const u8) !Array {
        const a: Array = .{ .ctx = l.src.tensor(name).ctx };
        if (l.quant.format == null or a.ctx == null or !narrowsLoadedF16(name, mlx.mlx_array_ndim(a), mlx.mlx_array_dtype(a))) return a;
        return l.own(try astypeEval(a, .bfloat16, l.s));
    }

    fn has(l: *const Loader, name: []const u8) bool {
        return l.src.tensor(name).ctx != null;
    }

    fn need(l: *Loader, name: []const u8) !Array {
        const a = try l.get(name);
        if (a.ctx == null) {
            log.err("MISSING WEIGHT: {s}\n", .{name});
            return error.MissingWeight;
        }
        return a;
    }

    /// `need`, kept.
    fn tensor(l: *Loader, name: []const u8) !Array {
        return l.keep(try l.need(name));
    }

    /// A release's e4m3 projection carries its 64 x 128 scale grid as `weight_scale_inv` (`fp8_block128`).
    fn triple(l: *Loader, base: []const u8) !Triple {
        return (try l.tripleOpt(base)) orelse {
            log.err("MISSING WEIGHT: {s}.weight\n", .{base});
            return error.MissingWeight;
        };
    }

    /// `triple`, or the pack's outside plugin's projection where the pack stores no `<base>.weight`; null: neither.
    fn tripleOpt(l: *Loader, base: []const u8) !?Triple {
        var nb: [256]u8 = undefined;
        const w = try l.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
        if (w.ctx == null) {
            const x = (try l.quant.linear(base, 0)) orelse return null;
            return .{ .w = x.w, .ext = x.ext };
        }
        const sc = try l.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}));
        return .{
            .w = w,
            .s = if (sc.ctx != null) sc else try l.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base})),
            .b = try l.get(try std.fmt.bufPrint(&nb, "{s}.biases", .{base})),
        };
    }

    /// A dense weight stored `[out, in]` as the `[in, out]` view a plain matmul reads.
    fn transposed(l: *Loader, t: *Triple) !void {
        if (t.s.ctx != null or t.w.ctx == null or t.ext != null) return;
        t.w = try l.own(try transposeLast(t.w, l.s));
    }

    fn kept(l: *Loader, t: Triple) !Triple {
        if (t.ext != null) return t;
        return .{ .w = try l.keep(t.w), .s = try l.keep(t.s), .b = try l.keep(t.b) };
    }

    fn lin(l: *Loader, raw: Triple, in_dim: ?u32) !Linear {
        if (raw.ext != null) return .{ .w = raw.w, .ext = raw.ext };
        // A pack an outside plugin claims keeps its plain matrices f16: run in the activation dtype, none promotes.
        if (l.quant.format != null and raw.s.ctx == null and raw.w.ctx != null and mlx.mlx_array_dtype(raw.w) == .float16 and mlx.getShape(raw.w).len >= 2)
            return .{ .w = try l.own(try astypeEval(raw.w, .bfloat16, l.s)) };
        const t = try l.kept(raw);
        if (t.s.ctx == null) return .{ .w = t.w };
        // e4m3: the matmul knows it by its dtypes; no affine kernel may claim it.
        if (fp8_block128.isBlockScaled(t.w, t.s)) return .{ .w = t.w, .s = t.s, .bits = 0, .group_size = fp8_block128.block_cols };
        const qp = resolve(l.c, t.w, t.s, in_dim);
        return .{ .w = t.w, .s = t.s, .b = t.b, .bits = qp.bits, .group_size = qp.group_size, .mode = qp.mode };
    }

    /// `<base>` as a projection over `in_dim` inputs (a dense one pre-transposed).
    fn proj(l: *Loader, base: []const u8, in_dim: ?u32) !Linear {
        var t = try l.triple(base);
        try l.transposed(&t);
        return l.lin(t, in_dim);
    }

    /// An f32 copy of `w` (transposed first when `transpose`), evaluated.
    fn f32Table(l: *Loader, w: Array, transpose: bool) !Array {
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        if (transpose) try mlx.check(mlx.mlx_transpose(&t, w, l.s)) else _ = mlx.mlx_array_set(&t, w);
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, t, .float32, l.s));
        const out = try l.own(try array.materializedOwnedCopy(l.s, f));
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    fn layerName(l: *const Loader, buf: []u8, i: u32, leaf: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ l.prefix, i, leaf });
    }

    fn mhc(l: *Loader, base: []const u8, comptime which: []const u8) !Mhc {
        var nb: [256]u8 = undefined;
        return .{
            .fn_rows = try l.f32Table(try l.need(try std.fmt.bufPrint(&nb, "{s}." ++ which ++ ".fn", .{base})), false),
            .base = try l.f32Table(try l.need(try std.fmt.bufPrint(&nb, "{s}." ++ which ++ ".base", .{base})), false),
            .scale = try l.f32Table(try l.need(try std.fmt.bufPrint(&nb, "{s}." ++ which ++ ".scale", .{base})), false),
        };
    }

    /// The sparse attention under `<base>.self_attn` (a trunk layer, or the MTP head's block).
    fn dsa(l: *Loader, base: []const u8) !Dsa {
        const c = l.c;
        var nb: [256]u8 = undefined;
        const at = struct {
            fn f(b: []u8, bs: []const u8, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(b, "{s}.self_attn.{s}", .{ bs, leaf });
            }
        }.f;
        return .{
            .q_a = try l.proj(try at(&nb, base, "q_a_proj"), c.hidden_size),
            .kv_a = try l.proj(try at(&nb, base, "kv_a_proj_with_mqa"), c.hidden_size),
            .q_a_norm = try l.tensor(try at(&nb, base, "q_a_layernorm.weight")),
            .kv_a_norm = try l.tensor(try at(&nb, base, "kv_a_layernorm.weight")),
            .q_b = try l.proj(try at(&nb, base, "q_b_proj"), c.dsa_q_lora_rank),
            .embed_q = try l.lin(try l.triple(try at(&nb, base, "embed_q")), c.dsa_head_dim),
            .unembed = try l.lin(try l.triple(try at(&nb, base, "unembed_out")), c.dsa_kv_lora_rank),
            .o = try l.proj(try at(&nb, base, "o_proj"), c.num_attention_heads * c.dsa_head_dim),
            .idx_wq_b = try l.proj(try at(&nb, base, "indexer.wq_b"), c.dsa_q_lora_rank),
            .idx_wk = try l.proj(try at(&nb, base, "indexer.wk"), c.hidden_size),
            .idx_k_norm_w = try l.tensor(try at(&nb, base, "indexer.k_norm.weight")),
            .idx_k_norm_b = try l.tensor(try at(&nb, base, "indexer.k_norm.bias")),
            .idx_wproj = try l.proj(try at(&nb, base, "indexer.weights_proj"), c.hidden_size),
            .idx_gate_t = try l.f32Table(try l.need(try at(&nb, base, "indexer.index_kpool_compress_gate")), true),
            .idx_ape = try l.f32Table(try l.need(try at(&nb, base, "indexer.index_kpool_compress_ape")), false),
        };
    }

    fn kda(l: *Loader, i: u32) !Kda {
        const c = l.c;
        const glm = c.isGlm5();
        var nb: [256]u8 = undefined;
        var lb: [128]u8 = undefined;
        const base = try std.fmt.bufPrint(&lb, "{s}.layers.{d}.{s}", .{ l.prefix, i, if (glm) "self_attn" else "attention" });
        // oMLX packs ship the three convs joined (`[C, K, 1]`) and the gate under `forget_gate.`.
        const fused_conv = try l.get(try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{base}));
        var fb: [160]u8 = undefined;
        const fg = if (l.has(try std.fmt.bufPrint(&nb, "{s}.forget_gate.A_log", .{base}))) try std.fmt.bufPrint(&fb, "{s}.forget_gate", .{base}) else base;
        var parts: [3]Triple = undefined;
        var conv_w: [3]Array = undefined;
        var any_scales = false;
        // An outside plugin's pack ships q|k|v as one projection (`qkv_proj`) beside the joined conv.
        const fused_qkv: ?Triple = if (l.quant.format != null and fused_conv.ctx != null) try l.tripleOpt(try std.fmt.bufPrint(&nb, "{s}.qkv_proj", .{base})) else null;
        if (fused_qkv == null) for (&parts, &conv_w, [_][]const u8{ "q", "k", "v" }) |*p, *cw, x| {
            p.* = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}_proj", .{ base, x }));
            if (p.s.ctx != null) any_scales = true;
            // The reference nests the conv in a module (`<x>_conv1d.conv.weight`); mlx-lm flattens it.
            if (fused_conv.ctx == null) {
                cw.* = try l.get(try std.fmt.bufPrint(&nb, "{s}.{s}_conv1d.conv.weight", .{ base, x }));
                if (cw.ctx == null) cw.* = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}_conv1d.weight", .{ base, x }));
            }
        };
        // q, k and v join only when quantized identically; otherwise each projects alone.
        const joinable = fused_qkv == null and blk: {
            var w_cols: [3]c_int = undefined;
            var s_cols: [3]c_int = undefined;
            for (parts, 0..) |p, n| {
                if (any_scales and p.s.ctx == null) break :blk false;
                w_cols[n] = lastDim(p.w);
                s_cols[n] = lastDim(p.s);
            }
            break :blk fused_rows.joinableCols(&w_cols) and fused_rows.joinableCols(&s_cols);
        };
        var qkv: Triple = fused_qkv orelse .{ .w = .{} };
        var qkv_parts: [3]Triple = @splat(.{ .w = .{} });
        if (joinable) {
            qkv.w = try l.own(try array.concatRows(&.{ parts[0].w, parts[1].w, parts[2].w }, l.s));
            if (any_scales) {
                qkv.s = try l.own(try array.concatRows(&.{ parts[0].s, parts[1].s, parts[2].s }, l.s));
                if (parts[0].b.ctx != null) qkv.b = try l.own(try array.concatRows(&.{ parts[0].b, parts[1].b, parts[2].b }, l.s));
            }
        } else if (fused_qkv == null) qkv_parts = parts;
        var joined_conv = if (fused_conv.ctx != null) fused_conv else try l.own(try array.concatRows(&conv_w, l.s));
        // HF-layout packs store the depthwise conv `[C, 1, K]`; MLX's conv1d reads `[C, K, 1]`.
        const csh = mlx.getShape(joined_conv);
        if (csh.len == 3 and csh[1] == 1 and csh[2] > 1) {
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_swapaxes(&t, joined_conv, 1, 2, l.s));
            joined_conv = try l.own(try array.materializedOwnedCopy(l.s, t));
            try mlx.check(mlx.mlx_array_eval(joined_conv));
        }
        // A release storing the conv in f32 runs it in the model's bf16, as the reference loads it.
        if (mlx.mlx_array_dtype(joined_conv) == .float32) joined_conv = try l.own(try astypeEval(joined_conv, .bfloat16, l.s));
        // A_log is per value head, the gate per key channel: expanded once so the hot path stays elementwise.
        const a_log_raw = try l.need(try std.fmt.bufPrint(&nb, "{s}.A_log", .{fg}));
        const dk: c_int = @intCast(c.linear_key_head_dim);
        const hv: c_int = @intCast(c.linear_num_value_heads);
        var col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col);
        try mlx.check(mlx.mlx_reshape(&col, a_log_raw, &[_]c_int{ hv, 1 }, 2, l.s));
        var bc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bc);
        try mlx.check(mlx.mlx_broadcast_to(&bc, col, &[_]c_int{ hv, dk }, 2, l.s));
        var a_log = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&a_log, bc, &[_]c_int{hv * dk}, 1, l.s));
        _ = try l.own(a_log);
        try mlx.check(mlx.mlx_array_eval(a_log));
        var z = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, if (glm) "g_a_proj" else "g_proj" }));
        var a = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ fg, if (glm) "f_a_proj" else "f_proj" }));
        var b = try l.triple(try std.fmt.bufPrint(&nb, "{s}.b_proj", .{base}));
        var o = try l.triple(try std.fmt.bufPrint(&nb, "{s}.o_proj", .{base}));
        var a2: Triple = .{ .w = .{} };
        var z2: Triple = .{ .w = .{} };
        if (glm) {
            a2 = try l.triple(try std.fmt.bufPrint(&nb, "{s}.f_b_proj", .{fg}));
            z2 = try l.triple(try std.fmt.bufPrint(&nb, "{s}.g_b_proj", .{base}));
        }
        const dt_bias = try l.need(try std.fmt.bufPrint(&nb, "{s}.dt_bias", .{fg}));
        var w: Kda = .{
            .z = undefined,
            .a = undefined,
            .b = undefined,
            .out = undefined,
            .conv_w = try l.keep(joined_conv),
            .A_log = a_log,
            .dt_bias = try l.keep(dt_bias),
            .norm_w = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.o_norm.weight", .{base})),
        };
        if (glm) {
            w.a_log_h = try l.own(try astypeEval(a_log_raw, .float32, l.s));
            w.dt_f32 = try l.own(try astypeEval(dt_bias, .float32, l.s));
        }
        for ([_]*Triple{ &qkv, &qkv_parts[0], &qkv_parts[1], &qkv_parts[2], &z, &a, &b, &o, &a2, &z2 }) |t| try l.transposed(t);
        // Decode reads q|k|v and the first gate stages as one row-joined matmul; the parts become views of it.
        if (glm and qkv.s.ctx != null) {
            const group = [_][3]*Array{ .{ &qkv.w, &qkv.s, &qkv.b }, .{ &z.w, &z.s, &z.b }, .{ &a.w, &a.s, &a.b }, .{ &b.w, &b.s, &b.b } };
            if (try fused_rows.fuseRowGroup(&group, l.s)) |f| {
                for ([_]Array{ f.w, f.s, f.b }) |x| _ = try l.own(x);
                for (group) |g| for (g) |x| {
                    _ = try l.own(x.*);
                };
                w.in = .{ .lin = try l.lin(.{ .w = f.w, .s = f.s, .b = f.b }, c.hidden_size), .widths = f.widths };
            }
        }
        w.qkv = try l.lin(qkv, c.hidden_size);
        for (&w.qkv_parts, qkv_parts) |*dst, t| dst.* = try l.lin(t, c.hidden_size);
        w.z = try l.lin(z, c.hidden_size);
        w.a = try l.lin(a, c.hidden_size);
        w.b = try l.lin(b, c.hidden_size);
        w.out = try l.lin(o, c.linear_num_value_heads * c.linear_value_head_dim);
        w.a2 = try l.lin(a2, c.linear_key_head_dim);
        w.z2 = try l.lin(z2, c.linear_key_head_dim);
        return w;
    }

    fn mla(l: *Loader, i: u32) !Mla {
        const c = l.c;
        var nb: [256]u8 = undefined;
        var w: Mla = .{
            .kv_a = try l.proj(try l.layerName(&nb, i, "attention.kv_a_proj_with_mqa"), c.hidden_size),
            .kv_a_norm = try l.tensor(try l.layerName(&nb, i, "attention.kv_a_layernorm.weight")),
            .kv_b = try l.proj(try l.layerName(&nb, i, "attention.kv_b_proj"), c.mla_kv_lora_rank),
            .g = try l.proj(try l.layerName(&nb, i, "attention.g_proj"), c.hidden_size),
            .o = try l.proj(try l.layerName(&nb, i, "attention.dense"), c.num_attention_heads * c.mla_v_head_dim),
        };
        // One query arm per checkpoint: the low-rank pair (tiny) or a direct q_proj (flash).
        if (c.mla_q_lora_rank > 0) {
            w.q_a = try l.proj(try l.layerName(&nb, i, "attention.q_a_proj"), c.hidden_size);
            w.q_a_norm = try l.tensor(try l.layerName(&nb, i, "attention.q_a_layernorm.weight"));
            w.q_b = try l.proj(try l.layerName(&nb, i, "attention.q_b_proj"), c.mla_q_lora_rank);
        } else w.q = try l.proj(try l.layerName(&nb, i, "attention.q_proj"), c.hidden_size);
        return w;
    }

    fn dense(l: *Loader, i: u32) !Dense {
        const c = l.c;
        var nb: [256]u8 = undefined;
        var g = try l.triple(try l.layerName(&nb, i, "mlp.gate_proj"));
        var u = try l.triple(try l.layerName(&nb, i, "mlp.up_proj"));
        var d = try l.triple(try l.layerName(&nb, i, "mlp.down_proj"));
        for ([_]*Triple{ &g, &u, &d }) |t| try l.transposed(t);
        var gu: linear.Joined = .{};
        if (g.s.ctx != null) {
            const group = [_][3]*Array{ .{ &g.w, &g.s, &g.b }, .{ &u.w, &u.s, &u.b } };
            if (try fused_rows.fuseRowGroup(&group, l.s)) |f| {
                for ([_]Array{ f.w, f.s, f.b }) |x| _ = try l.own(x);
                for (group) |p| for (p) |x| {
                    _ = try l.own(x.*);
                };
                gu = .{ .lin = try l.lin(.{ .w = f.w, .s = f.s, .b = f.b }, c.hidden_size), .widths = f.widths };
            }
        }
        return .{ .gate = try l.lin(g, c.hidden_size), .up = try l.lin(u, c.hidden_size), .down = try l.lin(d, c.intermediate_size), .gu = gu };
    }

    /// The routed MoE under `<base>.mlp` (a trunk layer, or the MTP head's block) as the shared block reads it: the
    /// sigmoid router with its selection bias, the ungated shared expert, ModelOpt's per-expert scales, a Sushi
    /// pack's EXL3 banks through the format table, GLM's one-kernel decode as the `fused` arm.
    /// `banked`: the layer's experts are the streaming pack's bank's, bound after the load (`bindBank`).
    fn routed(l: *Loader, base: []const u8, banked: bool) !moe.Moe.Weights {
        const c = l.c;
        const glm = c.isGlm5();
        var nb: [256]u8 = undefined;
        // mlx-lm quantizes the router under its module (`mlp.gate.gate_proj`); the reference keeps `mlp.gate`.
        const rt = if (!l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.weight", .{base})) and l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.gate_proj.weight", .{base}))) "mlp.gate.gate_proj" else "mlp.gate";
        const ex = if (l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.experts.gate_proj.weight", .{base}))) "mlp.experts" else "mlp.switch_mlp";
        const spec: ?exl3.Spec = if (glm) l.exl3 else null;
        const lf: [3][]const u8 = if (spec != null) .{ "trellis", "suh", "svh" } else .{ "weight", "scales", "biases" };
        var banks: [3]Triple = @splat(.{ .w = .{} });
        // An outside plugin's resident experts (a streaming pack's head) run as the block's format banks.
        const outside_banks: ?moe.FormatBanks = if (banked or l.has(try std.fmt.bufPrint(&nb, "{s}.{s}.gate_proj.weight", .{ base, ex }))) null else try l.quant.experts(try std.fmt.bufPrint(&nb, "{s}.mlp.experts", .{base}), c.hidden_size, c.moe_intermediate_size);
        if (!banked and outside_banks == null) for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*bk, p| {
            bk.* = .{
                .w = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[0] })),
                .s = try l.get(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[1] })),
                .b = try l.get(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[2] })),
            };
        };
        var router = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, rt }));
        var shared: [3]Triple = undefined;
        for (&shared, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*t, p| t.* = try l.triple(try std.fmt.bufPrint(&nb, "{s}.mlp.shared_experts.{s}", .{ base, p }));
        const expert_bias = if (glm)
            try l.f32Table(try l.need(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.e_score_correction_bias", .{base})), false)
        else blk: {
            const eb = try l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.expert_bias", .{base}));
            if (eb.ctx != null) break :blk try l.keep(eb);
            // A pack without the bias routes on plain sigmoid scores (the reference zero-inits it).
            var zeros = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{@intCast(c.num_experts)}, 1, .float32, l.s));
            break :blk try l.own(zeros);
        };
        const per_expert = try l.keep(try l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.expert_scale", .{base})));
        const gate_scale = try l.keep(try l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.gate_proj.expert_scale", .{base})));
        const up_scale = try l.keep(try l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.up_proj.expert_scale", .{base})));
        try l.transposed(&router);
        // GLM routes in f32 (`moe_router_dtype`).
        if (glm and router.s.ctx == null) router.w = try l.f32Table(router.w, false);
        var pb: [112]u8 = undefined;
        const mlp_prefix = try std.fmt.bufPrint(&pb, "{s}.mlp", .{base});
        // The head's experts are no routed layer of the stream: they stay resident.
        const stream: ?*streamed.Streamed = if (l.experts) |x| (if (x.layerOf(mlp_prefix) != null) x else null) else null;
        // A streamed layer's banks stay the source's, read for their quant params only: one the model kept would be
        // made resident by `evalTrunk`.
        if (stream == null) for (&banks) |*t| {
            try l.transposed(t);
            t.* = try l.kept(t.*);
        };
        for (&shared) |*t| try l.transposed(t);
        if (spec) |sp| for (banks, [_][2]u32{ .{ c.hidden_size, c.moe_intermediate_size }, .{ c.hidden_size, c.moe_intermediate_size }, .{ c.moe_intermediate_size, c.hidden_size } }) |t, io|
            try exl3.admit(.{ .trellis = t.w, .suh = t.s, .svh = t.b }, c.num_experts, io[0], io[1], sp);

        const inter: ?u32 = if (c.moe_intermediate_size > 0) c.moe_intermediate_size else null;
        const shared_inter: ?u32 = if (c.shared_expert_intermediate_size > 0) c.shared_expert_intermediate_size else null;
        var w: moe.Moe.Weights = .{
            .router = try l.lin(router, c.hidden_size),
            .route = .{ .sigmoid_bias = .{
                .expert_bias = expert_bias,
                .norm = c.moe_route_norm,
                .scale = c.router_scaling_factor,
                .n_group = @intCast(@max(c.moe_n_group, 1)),
                .topk_group = @intCast(@max(c.moe_topk_group, 1)),
            } },
            .gate = bankOf(c, banks[0], c.hidden_size, gate_scale),
            .up = bankOf(c, banks[1], c.hidden_size, up_scale),
            .down = bankOf(c, banks[2], inter, .{}),
            .shared = .{
                .gate = try l.lin(shared[0], c.hidden_size),
                .up = try l.lin(shared[1], c.hidden_size),
                .down = try l.lin(shared[2], shared_inter),
            },
            .top_k = @intCast(c.num_experts_per_tok),
            .experts = @intCast(c.num_experts),
            .clamp = clampOf(c),
            .clamp_limit = if (c.swiglu_clamp) c.swiglu_limit else 0,
            .route_scale = per_expert,
        };
        if (stream) |x| {
            try moe.bindStreamed(&w, x, mlp_prefix);
        } else if (outside_banks) |fb| {
            w.format_banks = fb;
        } else if (banked) {
            // Bound to the bank after the load.
        } else if (spec) |sp| {
            w.format_banks = .{ .table = &exl3.table, .gate = exl3Weight(banks[0], sp), .up = exl3Weight(banks[1], sp), .down = exl3Weight(banks[2], sp) };
        } else if (glm) w.fused = .{ .rows = glm5.MOE_DECODE_MAX_ROWS, .limit = c.swiglu_limit, .run = &@import("moe.zig").fusedDecode };
        w.route_fn = moe.compileRouting(&w, l.s);
        return w;
    }

    /// `<root>.mtp.0`, the head beside the trunk's `.model`; null when the pack ships none.
    /// Ling's NextN layer at `num_hidden_layers`; null when the pack ships none (no `eh_proj`). Its routed layer stays
    /// resident (`banked` false): a streaming pack's experts load as the claiming plugin's format banks.
    fn lingHead(l: *Loader) !?LingHead {
        const c = l.c;
        const idx: u32 = c.num_hidden_layers;
        var bb: [112]u8 = undefined;
        const base = try std.fmt.bufPrint(&bb, "{s}.layers.{d}", .{ l.prefix, idx });
        var nb: [256]u8 = undefined;
        var eh = (try l.tripleOpt(try std.fmt.bufPrint(&nb, "{s}.eh_proj", .{base}))) orelse return null;
        try l.transposed(&eh);
        return .{
            .enorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.enorm.weight", .{base})),
            .hnorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.hnorm.weight", .{base})),
            .final_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.final_layernorm.weight", .{base})),
            .input_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.input_layernorm.weight", .{base})),
            .post_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.post_attention_layernorm.weight", .{base})),
            .eh_proj = try l.lin(eh, 2 * c.hidden_size),
            .mla = try l.mla(idx),
            .moe = try l.routed(base, false),
        };
    }

    fn head(l: *Loader) !?Head {
        const c = l.c;
        var bb: [96]u8 = undefined;
        const root = if (std.mem.endsWith(u8, l.prefix, ".model")) l.prefix[0 .. l.prefix.len - ".model".len] else l.prefix;
        const base = try std.fmt.bufPrint(&bb, "{s}.mtp.0", .{root});
        var nb: [256]u8 = undefined;
        var eh = (try l.tripleOpt(try std.fmt.bufPrint(&nb, "{s}.eh_proj", .{base}))) orelse return null;
        try l.transposed(&eh);
        var kb: [112]u8 = undefined;
        const block = try std.fmt.bufPrint(&kb, "{s}.block", .{base});
        return .{
            .enorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.enorm.weight", .{base})),
            .hnorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.hnorm.weight", .{base})),
            .norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{base})),
            .input_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.input_layernorm.weight", .{block})),
            .post_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.post_attention_layernorm.weight", .{block})),
            .eh_proj = try l.lin(eh, 2 * c.hidden_size),
            .dsa = try l.dsa(block),
            .moe = try l.routed(block, false),
        };
    }

    /// Every layer, in order; on a failure the built ones' compiled routing goes with the slice.
    fn layers(l: *Loader) ![]Layer {
        const c = l.c;
        const out = try gpa.alloc(Layer, c.num_hidden_layers);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*x| if (x.mlp == .moe) compiled.free(&x.mlp.moe.route_fn);
            gpa.free(out);
        }
        var nb: [256]u8 = undefined;
        for (out, 0..) |*L, li| {
            const i: u32 = @intCast(li);
            var lb: [96]u8 = undefined;
            const base = try std.fmt.bufPrint(&lb, "{s}.layers.{d}", .{ l.prefix, i });
            L.* = .{
                .input_norm = try l.tensor(try l.layerName(&nb, i, "input_layernorm.weight")),
                .post_norm = try l.tensor(try l.layerName(&nb, i, "post_attention_layernorm.weight")),
                .mixer = undefined,
                .mlp = undefined,
            };
            if (c.isGlm5()) {
                L.attn_hc = try l.mhc(base, "attn_hc");
                L.ffn_hc = try l.mhc(base, "ffn_hc");
            }
            L.mixer = if (c.isLinearLayer(i)) .{ .kda = try l.kda(i) } else if (c.isGlm5()) .{ .dsa = try l.dsa(base) } else .{ .mla = try l.mla(i) };
            L.mlp = if (c.num_experts > 0 and i >= c.first_k_dense_replace) .{ .moe = try l.routed(base, c.streamed_bank) } else .{ .dense = try l.dense(i) };
            if (L.mlp == .moe and i < c.expert_limit_out.len) {
                // Ling-3.1's clamps after the SiLU: the shared expert's in the block; the routed experts' in the composed
                // chain (no kernel serves it), or in the bound bank's own arithmetic (`BankGeometry.limits_out`, `bindBank`).
                L.mlp.moe.shared_limit_out = @floatFromInt(c.shared_limit_out[i]);
                if (c.expert_limit_out[i] != 0) {
                    L.mlp.moe.limit_out = @floatFromInt(c.expert_limit_out[i]);
                    // the composed chain's arm: no kernel claims the layer (a format bank takes `Swiglu.limit_out`)
                    if (!c.streamed_bank and L.mlp.moe.format_banks == null) L.mlp.moe.clamp = null;
                }
            }
            built += 1;
        }
        return out;
    }
};

/// A routed bank as the shared block reads it (a dense one at the pack's declared params).
fn bankOf(c: *const Config, t: Triple, in_dim: ?u32, scale: Array) moe.Bank {
    return .{ .w = t.w, .s = t.s, .b = t.b, .qp = resolve(c, t.w, t.s, in_dim), .expert_scale = scale };
}

fn lastDim(x: Array) c_int {
    if (x.ctx == null) return 0;
    const sh = mlx.getShape(x);
    return if (sh.len == 0) 0 else sh[sh.len - 1];
}

fn astypeEval(x: Array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The whole-number SwiGLU clamp the fused epilogues take: 0 for none, null for one no kernel serves.
fn clampOf(c: *const Config) ?u32 {
    if (c.swiglu_limit <= 0.0) return 0;
    if (!c.swiglu_clamp or @round(c.swiglu_limit) != c.swiglu_limit or c.swiglu_limit > 1 << 16) return null;
    return @intFromFloat(c.swiglu_limit);
}

/// A Sushi bank as the EXL3 format table loads one: trellis, suh, svh and the pack's decode.
fn exl3Weight(t: Triple, spec: exl3.Spec) quant_abi.Weight {
    const d = exl3.decode(spec);
    return .{
        .w = .{ .ctx = t.w.ctx },
        .scales = .{ .ctx = t.s.ctx },
        .biases = .{ .ctx = t.b.ctx },
        .layout = @backingInt(d.codebook),
        .flags = @intCast(d.window.bits()),
        .block = @intCast(mlx.getShape(t.w)[3]),
        .state = if (d.table) |i| @ptrFromInt(@as(usize, i) + 1) else null,
    };
}

/// The weight prefix the pack uses: the config's, else the other nesting that holds the embedding.
fn probePrefix(src: *const quant_abi.Source, declared: []const u8, leaf: []const u8) []const u8 {
    var nb: [160]u8 = undefined;
    const candidates = [_][]const u8{ declared, "language_model.model", "model.language_model", "model" };
    for (candidates) |p| {
        const name = std.fmt.bufPrint(&nb, "{s}.{s}", .{ p, leaf }) catch continue;
        if (src.tensor(name).ctx != null) return p;
    }
    return declared;
}

/// The trunk from `ctx.weights`, and GLM's head when the pack ships one (a head that fails to load is dropped).
pub fn load(ctx: *const abi.LoadCtx) !*Model {
    const src = ctx.weights orelse return error.UnsupportedWeightSource;
    const c = config.of(ctx.parsed);
    // Before the trunk: what is outside the server, for the bank's open (`Fit.load_start`).
    const start = outside_model.readMemory();
    const s: mlx.mlx_stream = if (ctx.stream.ctx) |p| .{ .ctx = p } else mlx.gpuStream();
    const emb_base: []const u8 = if (std.mem.eql(u8, c.model_type, "bailing_hybrid")) "word_embeddings" else "embed_tokens";
    var eb: [64]u8 = undefined;
    const prefix = probePrefix(src, c.weight_prefix, try std.fmt.bufPrint(&eb, "{s}.weight", .{emb_base}));
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .c = c, .s = s, .caps = ctx.parsed.caps, .embed = undefined, .head = undefined, .final_norm = undefined };
    errdefer m.deinit();
    m.quant = try outside_model.Load.open(gpa, ctx, src.*, s);
    // The table is the load's: the weights it loaded stay the model's.
    defer m.quant.close();
    if (c.streamed_bank and m.quant.format == null) return error.StreamedBankUnclaimed;
    var l: Loader = .{ .src = src, .c = c, .s = s, .prefix = prefix, .owned = &m.owned, .exl3 = null, .quant = &m.quant };
    if (c.num_experts > 0 and !c.streamed_bank) m.experts = try streamed.Streamed.forLoad(gpa, ctx, c.num_experts, s);
    if (m.experts) |*x| l.experts = x;
    if (c.expert_quant_json.len > 0) {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const text = try std.fmt.allocPrint(arena.allocator(), "{{\"expert_quant\":{s}}}", .{c.expert_quant_json});
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{});
        l.exl3 = try exl3.parse(root.object, c.num_experts_per_tok);
    }
    var nb: [256]u8 = undefined;
    m.rms_eps = mlx.mlx_array_new_float(c.rms_norm_eps);
    m.kda.gate_fn = gdn_base.compileGate(m.ssmStateDtype(), s);

    // The table's dtype decides dense, never the pack's declared bits; biases may be absent on a bias-less mode.
    const bias_mandatory = c.quant_bits > 0 and c.quant_mode == .affine;
    var emb = Triple{ .w = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}.weight", .{ prefix, emb_base })) };
    if (!floatDtype(emb.w)) {
        emb.s = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}.scales", .{ prefix, emb_base }));
        const bn = try std.fmt.bufPrint(&nb, "{s}.{s}.biases", .{ prefix, emb_base });
        emb.b = if (bias_mandatory) try l.need(bn) else try l.get(bn);
    }
    m.final_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}));
    const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
    var head: Triple = .{ .w = try l.get(try std.fmt.bufPrint(&nb, "{s}.lm_head.weight", .{lm_prefix})) };
    if (head.w.ctx != null) {
        if (!floatDtype(head.w)) {
            head.s = try l.need(try std.fmt.bufPrint(&nb, "{s}.lm_head.scales", .{lm_prefix}));
            const bn = try std.fmt.bufPrint(&nb, "{s}.lm_head.biases", .{lm_prefix});
            head.b = if (bias_mandatory) try l.need(bn) else try l.get(bn);
        }
    } else if (l.has("lm_head.weight")) {
        head = .{ .w = try l.get("lm_head.weight"), .s = try l.get("lm_head.scales"), .b = try l.get("lm_head.biases") };
        if (head.s.ctx == null) head.s = emb.s;
        if (head.b.ctx == null) head.b = emb.b;
    } else if (try l.tripleOpt("lm_head")) |t| {
        head = t;
    } else if (c.tie_word_embeddings) {
        head = emb;
    } else {
        log.err("MISSING WEIGHT: lm_head.weight\n", .{});
        return error.MissingWeight;
    }
    m.embed = try l.lin(emb, c.hidden_size);
    m.head = try l.lin(head, c.hidden_size);
    m.layers = try l.layers();
    try evalTrunk(m);
    if (c.isGlm5()) m.mtp = l.head() catch |e| blk: {
        log.warn("GLM MTP head: {s} — MTP off\n", .{@errorName(e)});
        break :blk null;
    };
    // Ling's NextN head on the streaming pack only: an MLX Ling pack loads as before.
    if (!c.isGlm5() and c.streamed_bank and c.nextn_layers > 0) m.ling_mtp = l.lingHead() catch |e| blk: {
        log.warn("Ling MTP head: {s} — MTP off\n", .{@errorName(e)});
        break :blk null;
    };
    // The drafts' readout of a head the outside plugin runs, before the bank takes its room: the engine borrows it at
    // the bind (`trunk_weight`).
    if ((m.mtp != null or m.ling_mtp != null) and m.head.ext != null) try draftReadout(m);
    if (c.streamed_bank) {
        // The head resident before the bank's open reads the memory (else lazy until `HeadHooks.load`, past the open).
        if (m.mtp != null) try readyHead(m);
        if (m.ling_mtp != null) try readyLingHead(m);
        try bindBank(m, ctx, start);
    }
    return m;
}

/// The streaming pack's bank, opened by its outside plugin at the host's room past the loaded residents, bound to every
/// routed layer (the trunk's layer index, as `BankGeometry.gates`).
/// The bank's per-phase fit (`outside_model.Fit`): the host's prompt and decode rooms, and the pack's measured
/// calibration (`bank-fit.json`) when it names this ceiling and reaches the server's context; the bills otherwise.
fn bankFit(ctx: *const abi.LoadCtx, start: outside_model.Reading) outside_model.Fit {
    const grow = ctx.streamGrowth();
    var fit: outside_model.Fit = .{ .prompt_room = ctx.stream_room, .decode_room = ctx.decodeRoom(), .grow_prompt = grow.prompt, .grow_decode = grow.decode, .load_start = start };
    if (!ctx.dir.has(outside_model.bank_fit_file)) return fit;
    var buf: [4096]u8 = undefined;
    var why: abi.Verdict = .ok;
    const text = ctx.dir.readFile(outside_model.bank_fit_file, &buf, &why) orelse {
        log.warn("glm5_next: {s} does not read ({s}): the bank fits by the bills\n", .{ outside_model.bank_fit_file, why.why.slice() });
        return fit;
    };
    fit.calibration = outside_model.parseBankFit(text, ctx.gpuCeiling(), abi.stream_room_cell) catch |e| {
        log.warn("glm5_next: {s} not applied ({s}): the bank fits by the bills\n", .{ outside_model.bank_fit_file, @errorName(e) });
        return fit;
    };
    log.info("glm5_next: the bank fits by {s} (ceiling {d} MiB; the bills' growth past its {d}-token evidence {d} / {d} MiB)\n", .{ outside_model.bank_fit_file, fit.calibration.?.ceiling >> 20, fit.calibration.?.evidence, grow.prompt >> 20, grow.decode >> 20 });
    return fit;
}

fn bindBank(m: *Model, ctx: *const abi.LoadCtx, start: outside_model.Reading) !void {
    const c = m.c;
    const gates = try gpa.alloc(quant_abi.BankGate, m.layers.len);
    defer gpa.free(gates);
    for (gates, m.layers) |*g, *ly| g.* = if (ly.mlp == .moe) .{ .w = .{ .ctx = ly.mlp.moe.router.w.ctx }, .bias = .{ .ctx = ly.mlp.moe.route.sigmoid_bias.expert_bias.ctx } } else .{};
    // Ling-3.1's routed clamps after the SiLU, per layer (null where no layer clamps so).
    const limits = try gpa.alloc(f32, m.layers.len);
    defer gpa.free(limits);
    var any_limit = false;
    for (limits, 0..) |*l, li| {
        l.* = if (li < c.expert_limit_out.len) @floatFromInt(c.expert_limit_out[li]) else 0;
        if (l.* > 0) any_limit = true;
    }
    const geo: quant_abi.BankGeometry = .{ .hidden = c.hidden_size, .inter = c.moe_intermediate_size, .top_k = c.num_experts_per_tok, .n_experts = c.num_experts, .n_layers = @intCast(m.layers.len), .swiglu_limit = if (c.swiglu_clamp) c.swiglu_limit else 0, .gates = gates.ptr, .limits_out = if (any_limit) limits.ptr else null };
    const o = try outside_model.openBank(m.quant.format.?, geo, .{ .fit = bankFit(ctx, start) }, m.s);
    m.bank = o.bank;
    log.info("glm5_next: streamed experts, {s}: {d} prompt / {d} decode slot rows per layer\n", .{ o.bank.b.about.slice(), o.rows.prompt, o.rows.decode });
    // the budget and the memory read at the open: a bank-fit.json's evidence (`rest = non-file in use - server footprint`)
    if (o.refused) |e| log.warn("glm5_next: the memory read at the open refused ({s}: non-file in use below the server's footprint): the bank fits by the bills on total - available\n", .{@errorName(e)});
    if (o.read) |r| if (r.in_use != o.reading.?.in_use) log.info("glm5_next: the open read non-file in use {d} MiB, settled to {d} MiB (outside the server at the load's start {d} MiB, server footprint {d} MiB)\n", .{ r.in_use >> 20, o.reading.?.in_use >> 20, (start.in_use -| start.footprint) >> 20, r.footprint >> 20 });
    if (o.reading) |r| log.info("glm5_next: bank budget {d} MiB prompt / {d} MiB decode: non-file in use {d} MiB, server footprint {d} MiB at the open\n", .{ o.budget.prompt >> 20, o.budget.decode >> 20, r.in_use >> 20, r.footprint >> 20 });
    const r = o.bank.routed();
    m.routed = r;
    for (m.layers, 0..) |*ly, li| if (ly.mlp == .moe) try moe.bindRouted(&ly.mlp.moe, r, @intCast(li));
}

/// Tells a streaming pack's bank which phase a forward of `rows` rows a call runs in: a call routing more ids than its
/// decode lane takes is a prompt pass.
pub fn bankPhase(m: *const Model, rows: usize) !void {
    const r = m.routed orelse return;
    try r.phase(rows * m.c.num_experts_per_tok > r.decode_ids);
}

fn floatDtype(a: Array) bool {
    return switch (mlx.mlx_array_dtype(a)) {
        .bfloat16, .float16, .float32 => true,
        else => false,
    };
}

/// Every trunk weight resident at load, as the legacy load evaluates its layers.
fn evalTrunk(m: *const Model) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (m.owned.items) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    try mlx.check(mlx.mlx_eval(vec));
    try m.quant.eval();
}

extern "c" fn mlx_eye(res: *Array, n: c_int, m: c_int, k: c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) c_int;

/// The draft readout of a head the outside plugin runs (`Model.draft_head`): its rows read once through the plugin
/// (`I @ W^T`, the identity's rows), stored as MLX affine 4-bit at group 64, the dense copy freed.
fn draftReadout(m: *Model) !void {
    const s = m.s;
    const h: c_int = @intCast(m.c.hidden_size);
    var eye = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eye);
    try mlx.check(mlx_eye(&eye, h, h, 0, .bfloat16, s));
    var eye3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eye3);
    try mlx.check(mlx.mlx_reshape(&eye3, eye, &[_]c_int{ 1, h, h }, 3, s));
    const wt = try m.mm(&m.head, eye3);
    defer _ = mlx.mlx_array_free(wt);
    const vocab = mlx.getShape(wt)[2];
    var wt2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt2);
    try mlx.check(mlx.mlx_reshape(&wt2, wt, &[_]c_int{ h, vocab }, 2, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_transpose(&w, wt2, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, w, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    const bits = readoutBits(m.c);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(@intCast(draft_head_group)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    var parts: [3]Array = @splat(mlx.mlx_array_new());
    for (&parts, 0..) |*a, i| {
        try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
        try m.owned.append(gpa, a.*);
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, 3);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    m.draft_head = .{ .w = parts[0], .s = parts[1], .b = parts[2], .bits = bits, .group_size = draft_head_group, .mode = .affine };
    var held: u64 = 0;
    for (parts) |a| held += mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
    const billed = draftReadoutBytes(@intCast(vocab), m.c.hidden_size, bits);
    log.info("MTP draft readout: the EXL3 head as affine {d}-bit g{d}, {d} MiB resident (billed {d} MiB)\n", .{ bits, draft_head_group, held >> 20, billed >> 20 });
    if (held != billed) return error.DraftReadoutUnbilled;
}

/// The draft readout's width: each model's evaluated composition (GLM-5.3-Flash re-scores on an 8-bit copy, Ling-3.1
/// on a 4-bit one). The engine shortlists on its own 3-bit copy of it and re-scores the shortlist on these rows.
pub fn readoutBits(c: *const Config) u32 {
    return if (c.isGlm5()) 8 else 4;
}
pub const draft_head_group: u32 = 64;

/// The draft readout's resident bytes at `vocab` x `hidden` and `bits` (`draftReadout`; the load's `resident_bytes`
/// bills them): the packed words and a bf16 scale and bias per group of 64. The engine bills its own shortlist head.
pub fn draftReadoutBytes(vocab: u64, hidden: u64, bits: u32) u64 {
    return vocab * hidden * bits / 8 + 2 * 2 * vocab * (hidden / draft_head_group);
}

test "glm5_next trunk: the draft readout bills Ling-3.1-flash's 4-bit head at 345 MiB, GLM-5.3-Flash's 8-bit head at 643 MiB" {
    try std.testing.expectEqual(@as(u64, 362_151_936), draftReadoutBytes(157_184, 4096, 4));
    try std.testing.expectEqual(@as(u64, 674_037_760), draftReadoutBytes(154_880, 4096, 8));
    try std.testing.expectEqual(@as(u32, 4), readoutBits(&.{ .model_type = "bailing_hybrid" }));
    try std.testing.expectEqual(@as(u32, 8), readoutBits(&.{ .model_type = "glm5_next" }));
}

/// Makes Ling's NextN layer resident (bailing_hybrid's `HeadHooks.load`).
pub fn readyLingHead(m: *Model) !void {
    if (m.mtp_ready) return;
    const h = m.ling_mtp orelse return error.NoMtpHead;
    try mlx.evalFields(h);
    try m.quant.eval();
    m.mtp_ready = true;
}

/// Makes the MTP head resident (`HeadHooks.load`).
pub fn readyHead(m: *Model) !void {
    if (m.mtp_ready) return;
    const h = m.mtp orelse return error.NoMtpHead;
    try mlx.evalFields(h);
    try m.quant.eval();
    m.mtp_ready = true;
}
