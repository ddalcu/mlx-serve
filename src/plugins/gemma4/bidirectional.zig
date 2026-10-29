//! Gemma trained as a bidirectional encoder (EmbeddingGemma on the Gemma 3 trunk, EmbeddingGemma 2 on Gemma 4's):
//! no cache, no causality. Full layers attend over every key, sliding layers within a symmetric band, both under the
//! batch's key mask (`Run.mask`); sandwich-normed attention and GeGLU, then (EmbeddingGemma 2) the per-layer input
//! block from the embeddings alone and the layer scalar. The forward returns the final-normed hidden rows; the
//! engine pools them and `project` maps pooled rows through the pack's projection head.
//! copied_from: plugins/gemma3/encoder.zig (`forwardGemma3EncoderWith`), engines/mlx/forward/encoder_head.zig
//! (`embedProjection`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const da = @import("../../sdk/bases/blocks/decoder_attention.zig");
const Projection = @import("../../sdk/bases/blocks/head.zig").Projection;
const encoderBandMask = @import("../../sdk/blocks/attention/masks.zig").encoderBandMask;
const gemma = @import("../gemma3/config.zig");
const load_mod = @import("../gemma3/load.zig");
const compiled_mod = @import("../gemma3/compiled.zig");
const decoder = @import("decoder.zig");

const b = bases.blocks;
const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = bases.Ctx;
const Linear = b.linear.Linear;
const Embed = b.embed.TableEmbed;
const Mlp = b.extras.GeGlu;
const Config = gemma.Config;
const gpa = load_mod.gpa;
const Encoder = @This();

pub const Run = run_mod.Run;
pub const Out = run_mod.Out;
pub const batches_decode = false;

pub const Attn = struct {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Array,
    k_norm: Array,
    /// The weight-less value norm (EmbeddingGemma 2); null = none.
    v_norm: Array = .{},
    geo: da.Geo,
    global: bool,
};

pub const Layer = struct {
    input_norm: Array,
    attn: Attn,
    post_attn_norm: Array,
    pre_ff_norm: Array,
    mlp: Mlp.Weights,
    post_ff_norm: Array,
    ple: ?decoder.Ple = null,
    layer_scalar: Array = .{},
};

pub const Weights = struct {
    embed: Embed.Weights,
    layers: []const Layer,
    final_norm: Array,
    /// The sliding layers' band: keys with `|i - j| < band` (0 = every layer attends over all keys).
    band: u32 = 0,
    /// Floor the band under the key mask at a finite value: a padded query whose band holds only padding would
    /// softmax over -inf and its NaN row reach the real rows through V (EmbeddingGemma 2).
    floor_pad: bool = false,
    /// The per-layer inputs' projection and norm (`ple_dim` 0 = none).
    ple_proj: Linear = .{},
    ple_proj_norm: Array = .{},
    ple_dim: c_int = 0,
    projection: ?Projection = null,
    compiled: *const compiled_mod.Compiled,
};

/// `q`, `k`, `v` of `x` normed and roped at 0..T-1, one bidirectional read under `mask`, the output projection.
fn attend(a: *const Attn, g: *Graph, x: Array, mask: Array, eps: f32) !Array {
    const geo = &a.geo;
    const sh = ops.shape(x);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const kv_shape = [_]c_int{ sh[0], sh[1], geo.kv_heads, geo.head_dim };
    const q = try g.reshape(try a.q.forward(g, x), &.{ sh[0], sh[1], geo.heads, geo.head_dim });
    const q_rope = try da.ropeAt(g, try g.transpose(try g.rmsNorm(q, a.q_norm, eps), &perm), geo, 0);
    const k_raw = try a.k.forward(g, x);
    const k = try g.reshape(k_raw, &kv_shape);
    const k_rope = try da.ropeAt(g, try g.transpose(try g.rmsNorm(k, a.k_norm, eps), &perm), geo, 0);
    var v = try g.reshape(if (a.v.w.ctx == a.k.w.ctx) k_raw else try a.v.forward(g, x), &kv_shape);
    if (a.v_norm.ctx != null) v = try g.rmsNorm(v, a.v_norm, eps);
    const out = try da.sdpaRaw(g, q_rope, k_rope, try g.transpose(v, &perm), geo.scale, if (mask.ctx != null) "array" else "", mask, false);
    return a.o.forward(g, try g.mergeHeads(out));
}

/// Every layer's per-layer input `[B, T, layers, ple_dim]`: `rms(project(h) * hidden^-0.5)`.
fn pleInput(w: *const Weights, g: *Graph, h: Array, eps: f32) !Array {
    const sh = ops.shape(h);
    const proj = try b.head.project(g, &w.ple_proj, h);
    const scaled = try g.mul(proj, try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(sh[2]))), ops.dtype(proj)));
    const n: c_int = @intCast(w.layers.len);
    return g.rmsNorm(try g.reshape(scaled, &.{ sh[0], sh[1], n, w.ple_dim }), w.ple_proj_norm, eps);
}

/// The hidden rows `[B, T, hidden]` of `ids` (`[B, T]`), media rows spliced at their placeholders.
pub fn forward(w: *const Weights, shared: *const bases.Shared, g: *Graph, ids: Array, run: Run) !Out {
    if (run.mode != .prefill) return error.EncoderPrefillOnly;
    var c = run.ctx(g, shared);
    const eps = shared.eps;
    var h = try run.splice(g, try Embed.forward(&w.embed, &c, ids), ids);
    const ple_input: Array = if (w.ple_dim > 0) try pleInput(w, g, h, eps) else .{};

    const t: usize = @intCast(ops.shape(h)[1]);
    const key_mask = run.mask;
    var sliding = key_mask;
    if (w.band > 0 and t > w.band) {
        const band = try g.adopt(try encoderBandMask(gpa, t, w.band, g.s));
        sliding = band;
        if (key_mask.ctx != null) {
            const sum = try g.add(band, key_mask);
            sliding = if (w.floor_pad) try g.maximum(sum, try g.scalar(-1.0e9, ops.dtype(sum))) else sum;
        }
    }

    for (w.layers, 0..) |*L, i| {
        var lg = g.scope();
        defer lg.deinit();
        c.g = &lg;
        c.layer = @intCast(i);
        const attn = try attend(&L.attn, &lg, try lg.rmsNorm(h, L.input_norm, eps), if (L.attn.global) key_mask else sliding, eps);
        var x = try lg.add(h, try lg.rmsNorm(attn, L.post_attn_norm, eps));
        const down = try Mlp.forward(&L.mlp, &c, try lg.rmsNorm(x, L.pre_ff_norm, eps));
        x = try lg.add(x, try lg.rmsNorm(down, L.post_ff_norm, eps));
        if (L.ple) |*p| x = try lg.add(x, try lg.rmsNorm(try decoder.pleProject(w.compiled, &c, p, x, ple_input), p.norm, eps));
        if (L.layer_scalar.ctx != null) x = try lg.mul(x, L.layer_scalar);
        c.g = g;
        const old = h;
        h = try g.keep(x);
        g.drop(old);
        try run.capture(g, @intCast(i), h);
        try run.cadence(i, w.layers.len, h);
    }
    return .{ .hidden = try g.rmsNorm(h, w.final_norm, eps) };
}

// ── load ──

/// Which encoder a pack is: EmbeddingGemma's (Gemma 3 trunk, `dense.0`/`dense.1` at the root) or EmbeddingGemma 2's
/// (per-layer inputs in a `ple_block`, layer scalars, an `embedding_projection` head).
pub const Kind = enum { embedding_gemma, embedding_gemma2 };

pub const Loaded = struct {
    base: bases.Model(Encoder),
    layers: []Layer = &.{},
    owned: load_mod.Owned = .{},
    compiled: compiled_mod.Compiled = undefined,

    fn release(m: *Loaded) void {
        m.compiled.deinit();
        m.owned.deinit();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

pub fn load(ctx: *const abi.LoadCtx, kind: Kind) !*Loaded {
    const c = gemma.of(ctx.parsed);
    var r = try load_mod.Reader.open(ctx, c);
    defer r.close();
    const s = r.s;
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids } };
    m.compiled.init(s, 0);
    errdefer m.release();
    m.layers = try gpa.alloc(Layer, c.num_hidden_layers);
    const o = &m.owned;
    const sh = try load_mod.AttnShared.build(o, c, s);
    const scale: f32 = if (kind == .embedding_gemma2) 1.0 else 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar)));
    const off = c.norm_has_offset;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        const li: u32 = @intCast(i);
        L.* = .{
            .input_norm = try r.norm(o, try r.name(&nb, "layers.{d}.input_layernorm.weight", .{li}), off),
            .attn = try attnAt(&r, o, c, &sh, li, scale),
            .post_attn_norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{li}), off),
            .pre_ff_norm = try r.norm(o, try r.name(&nb, "layers.{d}.pre_feedforward_layernorm.weight", .{li}), off),
            .mlp = try load_mod.mlpAt(&r, o, c, li, false, &m.compiled),
            .post_ff_norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_feedforward_layernorm.weight", .{li}), off),
        };
        if (kind == .embedding_gemma2) {
            L.layer_scalar = try r.need(try r.name(&nb, "layers.{d}.layer_scalar", .{li}));
            try o.own(L.layer_scalar);
            const pd = c.hidden_size_per_layer_input;
            L.ple = .{
                .gate = try ownedProj(&r, o, try r.name(&nb, "layers.{d}.ple_block.per_layer_input_gate", .{li}), c.hidden_size),
                .proj = try ownedProj(&r, o, try r.name(&nb, "layers.{d}.ple_block.per_layer_projection", .{li}), pd),
                .norm = try r.norm(o, try r.name(&nb, "layers.{d}.ple_block.post_per_layer_input_norm.weight", .{li}), false),
            };
        }
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try o.linear(embed);
    const emb_scale = load_mod.bf16Scalar(@sqrt(@as(f32, @floatFromInt(c.hidden_size))), s);
    try o.own(emb_scale);
    var w: Weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = .bfloat16, .scale = emb_scale },
        .layers = m.layers,
        .final_norm = try r.norm(o, try r.name(&nb, "norm.weight", .{}), off),
        .band = if (!c.has_sliding_window) 0 else if (kind == .embedding_gemma2) c.sliding_window + 1 else c.sliding_window,
        .floor_pad = kind == .embedding_gemma2,
        .compiled = &m.compiled,
    };
    switch (kind) {
        .embedding_gemma => if (r.has("dense.0.weight")) {
            const first = try ownedTable(&r, o, "dense.0", c.hidden_size);
            w.projection = .{ .first = first, .second = if (r.has("dense.1.weight")) try ownedTable(&r, o, "dense.1", @intCast(mlx.getShape(first.w)[0])) else null };
        },
        .embedding_gemma2 => {
            w.projection = .{ .first = try ownedTable(&r, o, try r.name(&nb, "embedding_projection", .{}), c.hidden_size) };
            w.ple_proj = try ownedTable(&r, o, try r.name(&nb, "ple.per_layer_model_projection", .{}), c.hidden_size);
            w.ple_proj_norm = try r.norm(o, try r.name(&nb, "ple.per_layer_projection_norm.weight", .{}), false);
            w.ple_dim = @intCast(c.hidden_size_per_layer_input);
        },
    }
    try mlx.check(mlx.mlx_array_eval(w.final_norm));
    m.base.weights = w;
    if (gemma.ternaryCandidate(c)) try load_mod.armTernary(.{ &m.layers, &m.base.weights.projection, &m.base.weights.ple_proj }, null, embed.w, s);
    return m;
}

fn ownedProj(r: *load_mod.Reader, o: *load_mod.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.proj(base, in_dim);
    try o.linear(l);
    return l;
}

/// A head kept in its checkpoint layout (`b.head.project` reads it).
fn ownedTable(r: *load_mod.Reader, o: *load_mod.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.tableAt(base, in_dim);
    try o.linear(l);
    return l;
}

/// Layer `li`'s attention: separate q/k/v/o (K = V on a full layer that ships no `v_proj`), the q/k norms, the
/// weight-less value norm where the family has one.
fn attnAt(r: *load_mod.Reader, o: *load_mod.Owned, c: *const Config, sh: *const load_mod.AttnShared, li: u32, scale: f32) !Attn {
    var nb: [256]u8 = undefined;
    const geo = load_mod.geoOf(c, sh, li, scale);
    const global = c.isGlobalLayer(li);
    const k_eq_v = c.attention_k_eq_v and global;
    const q = try ownedProj(r, o, try r.name(&nb, "layers.{d}.self_attn.q_proj", .{li}), c.hidden_size);
    const k = try ownedProj(r, o, try r.name(&nb, "layers.{d}.self_attn.k_proj", .{li}), c.hidden_size);
    const v = if (k_eq_v) k else try ownedProj(r, o, try r.name(&nb, "layers.{d}.self_attn.v_proj", .{li}), c.hidden_size);
    const dual = c.dualHeadDim() and global;
    return .{
        .q = q,
        .k = k,
        .v = v,
        .o = try ownedProj(r, o, try r.name(&nb, "layers.{d}.self_attn.o_proj", .{li}), @intCast(geo.heads * geo.head_dim)),
        .q_norm = try r.norm(o, try r.name(&nb, "layers.{d}.self_attn.q_norm.weight", .{li}), c.norm_has_offset),
        .k_norm = try r.norm(o, try r.name(&nb, "layers.{d}.self_attn.k_norm.weight", .{li}), c.norm_has_offset),
        .v_norm = if (!c.has_v_norm) .{} else if (dual and sh.v_norm_global.ctx != null) sh.v_norm_global else sh.v_norm,
        .geo = geo,
        .global = global,
    };
}

// ── the table's entry points ──

pub fn loadAs(comptime kind: Kind) *const fn (*const abi.LoadCtx, *abi.Verdict) callconv(.c) ?*anyopaque {
    return &struct {
        fn f(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
            return load(ctx, kind) catch |e| {
                out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
                return null;
            };
        }
    }.f;
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

/// Prefill only (decode, verify and batched rows refused `unsupported`); the hidden rows, no logits.
pub fn tableForward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    if (!abi.declaresMode(.{ .embed = true }, f.mode)) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Encoder, &m.base, f, out);
}

/// Pooled rows `[B, hidden]` through the projection head, f32; the pooled rows themselves without one.
pub fn project(mp: *anyopaque, pooled: abi.Array, out: *abi.Array) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    var g = Graph.init(gpa, m.base.s);
    defer g.deinit();
    const x: Array = .{ .ctx = pooled.ctx };
    const y = if (m.base.weights.projection) |*p| p.forward(&g, x) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) } else x;
    var h = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&h, y);
    out.* = .{ .ctx = h.ctx };
    return .ok;
}

/// No KV: an encoder's activations are its whole bill, the engine's default arithmetic.
pub fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}
