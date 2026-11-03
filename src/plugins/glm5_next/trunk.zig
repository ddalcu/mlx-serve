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
const build_options = @import("build_options");
pub const Bank = if (build_options.mlx_stream) @import("../../sdk/weights/streamed.zig").Glm5n.Source else void;
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
    /// A streamed pack's router as stored (bf16 `[experts, hidden]`): the bank's lookahead predictor reads it.
    router_raw: Array = .{},
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
    kda: KdaConsts = .{},
    /// `rms_norm_eps` as an array (the add+norm kernel's).
    rms_eps: Array = .{},
    owned: std.ArrayList(Array) = .empty,
    /// A streamed pack's expert bank, bound to every routed layer after the load (`plugin.zig` `openBank`).
    bank: ?*Bank = null,

    pub fn deinit(m: *Model) void {
        if (comptime build_options.mlx_stream) if (m.bank) |b| b.deinit();
        m.bank = null;
        for (m.layers) |*l| if (l.mlp == .moe) compiled.free(&l.mlp.moe.route_fn);
        if (m.mtp) |*h| compiled.free(&h.moe.route_fn);
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

    /// `x @ l^T` at `l`'s own quant geometry.
    pub fn mm(m: *const Model, l: *const Linear, x: Array) !Array {
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
        if (m.head.s.ctx != null) return m.mm(&m.head, x);
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
        if (dw.gu.built() and rows <= fused_rows.FUSED_ROWS_MAX_M) {
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

/// A weight triple as stored (`<base>.{weight,scales,biases}`), scales and biases absent on a dense weight.
const Triple = struct { w: Array, s: Array = .{}, b: Array = .{} };

const Loader = struct {
    src: *const quant_abi.Source,
    c: *const Config,
    s: mlx.mlx_stream,
    prefix: []const u8,
    owned: *std.ArrayList(Array),
    exl3: ?exl3.Spec,

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

    /// The source's tensor `name`, borrowed for the load; empty when the pack has none.
    fn get(l: *const Loader, name: []const u8) Array {
        return .{ .ctx = l.src.tensor(name).ctx };
    }

    fn has(l: *const Loader, name: []const u8) bool {
        return l.get(name).ctx != null;
    }

    fn need(l: *const Loader, name: []const u8) !Array {
        const a = l.get(name);
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
    fn triple(l: *const Loader, base: []const u8) !Triple {
        var nb: [256]u8 = undefined;
        const sc = l.get(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}));
        return .{
            .w = try l.need(try std.fmt.bufPrint(&nb, "{s}.weight", .{base})),
            .s = if (sc.ctx != null) sc else l.get(try std.fmt.bufPrint(&nb, "{s}.weight_scale_inv", .{base})),
            .b = l.get(try std.fmt.bufPrint(&nb, "{s}.biases", .{base})),
        };
    }

    /// A dense weight stored `[out, in]` as the `[in, out]` view a plain matmul reads.
    fn transposed(l: *Loader, t: *Triple) !void {
        if (t.s.ctx != null or t.w.ctx == null) return;
        t.w = try l.own(try transposeLast(t.w, l.s));
    }

    fn kept(l: *Loader, t: Triple) !Triple {
        return .{ .w = try l.keep(t.w), .s = try l.keep(t.s), .b = try l.keep(t.b) };
    }

    fn lin(l: *Loader, raw: Triple, in_dim: ?u32) !Linear {
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
        const fused_conv = l.get(try std.fmt.bufPrint(&nb, "{s}.conv1d.weight", .{base}));
        var fb: [160]u8 = undefined;
        const fg = if (l.has(try std.fmt.bufPrint(&nb, "{s}.forget_gate.A_log", .{base}))) try std.fmt.bufPrint(&fb, "{s}.forget_gate", .{base}) else base;
        var parts: [3]Triple = undefined;
        var conv_w: [3]Array = undefined;
        var any_scales = false;
        for (&parts, &conv_w, [_][]const u8{ "q", "k", "v" }) |*p, *cw, x| {
            p.* = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}_proj", .{ base, x }));
            if (p.s.ctx != null) any_scales = true;
            // The reference nests the conv in a module (`<x>_conv1d.conv.weight`); mlx-lm flattens it.
            if (fused_conv.ctx == null) {
                cw.* = l.get(try std.fmt.bufPrint(&nb, "{s}.{s}_conv1d.conv.weight", .{ base, x }));
                if (cw.ctx == null) cw.* = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}_conv1d.weight", .{ base, x }));
            }
        }
        // q, k and v join only when quantized identically; otherwise each projects alone.
        const joinable = blk: {
            var w_cols: [3]c_int = undefined;
            var s_cols: [3]c_int = undefined;
            for (parts, 0..) |p, n| {
                if (any_scales and p.s.ctx == null) break :blk false;
                w_cols[n] = lastDim(p.w);
                s_cols[n] = lastDim(p.s);
            }
            break :blk fused_rows.joinableCols(&w_cols) and fused_rows.joinableCols(&s_cols);
        };
        var qkv: Triple = .{ .w = .{} };
        var qkv_parts: [3]Triple = @splat(.{ .w = .{} });
        if (joinable) {
            qkv.w = try l.own(try array.concatRows(&.{ parts[0].w, parts[1].w, parts[2].w }, l.s));
            if (any_scales) {
                qkv.s = try l.own(try array.concatRows(&.{ parts[0].s, parts[1].s, parts[2].s }, l.s));
                if (parts[0].b.ctx != null) qkv.b = try l.own(try array.concatRows(&.{ parts[0].b, parts[1].b, parts[2].b }, l.s));
            }
        } else qkv_parts = parts;
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
    /// `streamed`: the layer's experts live in the pack's bank (bound after the load), not the shards.
    fn routed(l: *Loader, base: []const u8, streamed: bool) !moe.Moe.Weights {
        const c = l.c;
        const glm = c.isGlm5();
        var nb: [256]u8 = undefined;
        // mlx-lm quantizes the router under its module (`mlp.gate.gate_proj`); the reference keeps `mlp.gate`.
        const rt = if (!l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.weight", .{base})) and l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.gate_proj.weight", .{base}))) "mlp.gate.gate_proj" else "mlp.gate";
        const ex = if (l.has(try std.fmt.bufPrint(&nb, "{s}.mlp.experts.gate_proj.weight", .{base}))) "mlp.experts" else "mlp.switch_mlp";
        const spec: ?exl3.Spec = if (glm) l.exl3 else null;
        const lf: [3][]const u8 = if (spec != null) .{ "trellis", "suh", "svh" } else .{ "weight", "scales", "biases" };
        // A streamed pack's experts live in its bank (`experts.bin`), bound after the load.
        var banks: [3]Triple = @splat(.{ .w = .{} });
        if (!streamed) for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*bk, p| {
            bk.* = .{
                .w = try l.need(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[0] })),
                .s = l.get(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[1] })),
                .b = l.get(try std.fmt.bufPrint(&nb, "{s}.{s}.{s}.{s}", .{ base, ex, p, lf[2] })),
            };
        };
        var router = try l.triple(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, rt }));
        var shared: [3]Triple = undefined;
        for (&shared, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |*t, p| t.* = try l.triple(try std.fmt.bufPrint(&nb, "{s}.mlp.shared_experts.{s}", .{ base, p }));
        const expert_bias = if (glm)
            try l.f32Table(try l.need(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.e_score_correction_bias", .{base})), false)
        else blk: {
            const eb = l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.gate.expert_bias", .{base}));
            if (eb.ctx != null) break :blk try l.keep(eb);
            // A pack without the bias routes on plain sigmoid scores (the reference zero-inits it).
            var zeros = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{@intCast(c.num_experts)}, 1, .float32, l.s));
            break :blk try l.own(zeros);
        };
        const per_expert = try l.keep(l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.expert_scale", .{base})));
        const gate_scale = try l.keep(l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.gate_proj.expert_scale", .{base})));
        const up_scale = try l.keep(l.get(try std.fmt.bufPrint(&nb, "{s}.mlp.switch_mlp.up_proj.expert_scale", .{base})));
        try l.transposed(&router);
        // GLM routes in f32 (`moe_router_dtype`).
        if (glm and router.s.ctx == null) router.w = try l.f32Table(router.w, false);
        if (!streamed) for (&banks) |*t| {
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
        if (spec) |sp| {
            w.format_banks = .{ .table = &exl3.table, .gate = exl3Weight(banks[0], sp), .up = exl3Weight(banks[1], sp), .down = exl3Weight(banks[2], sp) };
        } else if (glm and !streamed) w.fused = .{ .rows = glm5.MOE_DECODE_MAX_ROWS, .limit = c.swiglu_limit, .run = &@import("moe.zig").fusedDecode };
        w.route_fn = moe.compileRouting(&w, l.s);
        return w;
    }

    /// `<root>.mtp.0`, the head beside the trunk's `.model`; null when the pack ships none.
    fn head(l: *Loader) !?Head {
        const c = l.c;
        var bb: [96]u8 = undefined;
        const root = if (std.mem.endsWith(u8, l.prefix, ".model")) l.prefix[0 .. l.prefix.len - ".model".len] else l.prefix;
        const base = try std.fmt.bufPrint(&bb, "{s}.mtp.0", .{root});
        var nb: [256]u8 = undefined;
        if (!l.has(try std.fmt.bufPrint(&nb, "{s}.eh_proj.weight", .{base}))) return null;
        var kb: [112]u8 = undefined;
        const block = try std.fmt.bufPrint(&kb, "{s}.block", .{base});
        return .{
            .enorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.enorm.weight", .{base})),
            .hnorm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.hnorm.weight", .{base})),
            .norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{base})),
            .input_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.input_layernorm.weight", .{block})),
            .post_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.post_attention_layernorm.weight", .{block})),
            .eh_proj = try l.proj(try std.fmt.bufPrint(&nb, "{s}.eh_proj", .{base}), 2 * c.hidden_size),
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
            L.mlp = if (c.num_experts > 0 and i >= c.first_k_dense_replace) .{ .moe = try l.routed(base, c.streamed) } else .{ .dense = try l.dense(i) };
            if (c.streamed and L.mlp == .moe) L.router_raw = try l.keep(try l.need(try l.layerName(&nb, i, "mlp.gate.weight")));
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
    const s: mlx.mlx_stream = if (ctx.stream.ctx) |p| .{ .ctx = p } else mlx.gpuStream();
    const emb_base: []const u8 = if (std.mem.eql(u8, c.model_type, "bailing_hybrid")) "word_embeddings" else "embed_tokens";
    var eb: [64]u8 = undefined;
    const prefix = probePrefix(src, c.weight_prefix, try std.fmt.bufPrint(&eb, "{s}.weight", .{emb_base}));
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .c = c, .s = s, .caps = ctx.parsed.caps, .embed = undefined, .head = undefined, .final_norm = undefined };
    errdefer m.deinit();
    var l: Loader = .{ .src = src, .c = c, .s = s, .prefix = prefix, .owned = &m.owned, .exl3 = null };
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
        emb.b = if (bias_mandatory) try l.need(bn) else l.get(bn);
    }
    m.final_norm = try l.tensor(try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{prefix}));
    const lm_prefix = if (std.mem.eql(u8, prefix, "language_model.model")) "language_model" else prefix;
    var head: Triple = .{ .w = l.get(try std.fmt.bufPrint(&nb, "{s}.lm_head.weight", .{lm_prefix})) };
    if (head.w.ctx != null) {
        if (!floatDtype(head.w)) {
            head.s = try l.need(try std.fmt.bufPrint(&nb, "{s}.lm_head.scales", .{lm_prefix}));
            const bn = try std.fmt.bufPrint(&nb, "{s}.lm_head.biases", .{lm_prefix});
            head.b = if (bias_mandatory) try l.need(bn) else l.get(bn);
        }
    } else if (l.has("lm_head.weight")) {
        head = .{ .w = l.get("lm_head.weight"), .s = l.get("lm_head.scales"), .b = l.get("lm_head.biases") };
        if (head.s.ctx == null) head.s = emb.s;
        if (head.b.ctx == null) head.b = emb.b;
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
    return m;
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
}

/// Makes the MTP head resident (`HeadHooks.load`).
pub fn readyHead(m: *Model) !void {
    if (m.mtp_ready) return;
    const h = m.mtp orelse return error.NoMtpHead;
    try mlx.evalFields(h);
    m.mtp_ready = true;
}
