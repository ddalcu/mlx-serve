//! A Gemma decoder's tensors out of the engine's loaded map, as the engine's loader binds them: each projection
//! through the pack's quant format, quantized q|k|v and gate|up joined into one buffer (originals freed), (1 + w)
//! norms folded where the family has them, the parameter-free value norms and the proportional rope table built, and
//! every attention layer's geometry resolved. Gemma 4 loads its layers through this file too.
//! copied_from: plugins/llama/plugin.zig (`Reader`, `join`, `headOf`), engines/mlx/load/layers_dense.zig
//! (`initStandardLayers`), engines/mlx/load/init.zig (embedding scale, v_norm ones, `rope_freqs_global`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const formats = @import("../../sdk/quant/formats.zig");
const native = @import("../../sdk/quant/native.zig");
const fuseRowGroup = @import("../../sdk/blocks/linear/fused_rows.zig").fuseRowGroup;
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
const config = @import("config.zig");

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Joined = b.linear.Joined;
const Config = config.Config;

/// The engine's ternary arm over a loaded Gemma model: the verdict over every quantized tensor it reads (an untied
/// embedding table skipped, as `firstNonTernaryBias`), qmv2's kernels for the 2-bit projections when it holds.
pub const armTernary = @import("../../sdk/bases/reader.zig").armTernary;
pub const gatherOnly = @import("../../sdk/bases/reader.zig").gatherOnly;

/// The Gemma attention block: sliding layers, KV sharing, value norms and K=V, the fused hd-256 decode step.
pub const Attention = b.decoder_attention.DecoderAttention(.{ .sliding = true, .kv_sharing = true, .value_norm = true });
const Geo = b.decoder_attention.Geo;

pub const gpa = std.heap.c_allocator;

fn ctxArr(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// Every array handle a loaded model holds, freed together.
pub const Owned = struct {
    list: std.ArrayList(Array) = .empty,

    pub fn own(o: *Owned, a: Array) !void {
        if (a.ctx == null) return;
        o.list.append(gpa, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
    }

    pub fn linear(o: *Owned, l: Linear) !void {
        for ([_]Array{ l.w, l.s, l.b, l.bias }) |a| try o.own(a);
    }

    pub fn deinit(o: *Owned) void {
        for (o.list.items) |a| _ = mlx.mlx_array_free(a);
        o.list.deinit(gpa);
    }
};

/// Where the tensors come from during one load.
pub const Reader = struct {
    src: *const quant_abi.Source,
    fmt: *const quant_abi.QuantTable,
    qctx: quant_abi.LoadCtx,
    prefix: []const u8,
    s: mlx.mlx_stream,

    /// Opens the pack's quant format over the engine's tensors; `close` when done.
    pub fn open(ctx: *const abi.LoadCtx, c: *const Config) !Reader {
        const src = ctx.weights orelse return error.NoWeightSource;
        const fmt = (formats.claimant(c.block_json, "{}") catch null) orelse &native.affine;
        var r: Reader = .{
            .src = src,
            .fmt = fmt,
            .qctx = .{ .block_json = abi.Str.of(c.block_json), .source = src, .stream = ctx.stream },
            .prefix = probePrefix(src, c.weight_prefix),
            .s = .{ .ctx = ctx.stream.ctx },
        };
        if (!fmt.open(&r.qctx).isOk()) return error.QuantFormatRefused;
        return r;
    }

    /// The configured prefix when the pack holds its embedding there, else the first mlx-lm layout that does (the
    /// engine's `resolveWeightPrefix`).
    fn probePrefix(src: *const quant_abi.Source, configured: []const u8) []const u8 {
        for ([_][]const u8{ configured, "language_model.model", "model.language_model", "model" }) |p| {
            var nb: [96]u8 = undefined;
            const key = std.fmt.bufPrint(&nb, "{s}.embed_tokens.weight", .{p}) catch continue;
            if (src.tensor(key).ctx != null) return p;
        }
        return configured;
    }

    pub fn close(r: *Reader) void {
        r.fmt.close(&r.qctx);
    }

    pub fn name(r: *const Reader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}." ++ fmt, .{r.prefix} ++ args);
    }

    pub fn has(r: *const Reader, full: []const u8) bool {
        return r.src.tensor(full).ctx != null;
    }

    /// A new handle on the source's tensor `full`, empty when the pack has none.
    pub fn share(r: *const Reader, full: []const u8) !Array {
        const a: Array = .{ .ctx = r.src.tensor(full).ctx };
        if (a.ctx == null) return a;
        var h = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_array_set(&h, a));
        return h;
    }

    pub fn need(r: *const Reader, full: []const u8) !Array {
        const a = try r.share(full);
        if (a.ctx == null) return error.MissingWeight;
        return a;
    }

    /// The projection `base` through the pack's quant format (`in_dim` pins its quant params), unowned.
    pub fn proj(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var w: quant_abi.Weight = .{};
        const v = r.fmt.load_tensor(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        if (w.layout == @backingInt(native.Layout.fp8_rows)) {
            r.fmt.release(&w);
            return error.RowScaledFp8Unsupported;
        }
        return .{ .w = ctxArr(w.w), .s = ctxArr(w.scales), .b = ctxArr(w.biases), .bits = w.bits, .group_size = w.group_size, .mode = native.modeOf(&w) };
    }

    /// A table kept in its checkpoint layout `[out, in]` (the embedding, the head): dense by its dtype, else through
    /// the format.
    pub fn tableAt(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var nb: [256]u8 = undefined;
        const w = r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
        if (w.ctx == null) return error.MissingWeight;
        switch (mlx.mlx_array_dtype(.{ .ctx = w.ctx })) {
            .bfloat16, .float16, .float32 => return .{ .w = try r.need(nb[0 .. base.len + 7]) },
            else => {},
        }
        if (r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.scales", .{base})).ctx == null) return error.MissingWeight;
        return r.proj(base, in_dim);
    }

    /// A norm weight, `1 + w` in bf16 where the family folds the offset (owned either way).
    pub fn norm(r: *const Reader, o: *Owned, full: []const u8, offset: bool) !Array {
        const raw = try r.need(full);
        if (!offset) {
            try o.own(raw);
            return raw;
        }
        defer _ = mlx.mlx_array_free(raw);
        const one = bf16Scalar(1.0, r.s);
        defer _ = mlx.mlx_array_free(one);
        var out = mlx.mlx_array_new();
        mlx.check(mlx.mlx_add(&out, one, raw, r.s)) catch |e| {
            _ = mlx.mlx_array_free(out);
            return e;
        };
        try o.own(out);
        return out;
    }
};

pub fn bf16Scalar(v: f32, s: mlx.mlx_stream) Array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, f, .bfloat16, s);
    return out;
}

/// `ones(n)` in bf16, owned.
pub fn ones(o: *Owned, n: u32, s: mlx.mlx_stream) !Array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_full(&out, &[_]c_int{@intCast(n)}, 1, one, .bfloat16, s)) catch |e| {
        _ = mlx.mlx_array_free(out);
        return e;
    };
    try o.own(out);
    return out;
}

/// Joins quantized projections reading one input into one buffer (rows) when they share a packed geometry, rewriting
/// each to a row view of it and freeing its own copy; empty when they do not.
pub fn join(o: *Owned, parts: []const *Linear, s: mlx.mlx_stream) !Joined {
    var triples: [4][3]*Array = undefined;
    var originals: [4][3]Array = undefined;
    for (parts, 0..) |p, i| {
        triples[i] = .{ &p.w, &p.s, &p.b };
        originals[i] = .{ p.w, p.s, p.b };
    }
    const fused = (try fuseRowGroup(triples[0..parts.len], s)) orelse return .{};
    for (originals[0..parts.len]) |orig| for (orig) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Array{ fused.w, fused.s, fused.b }) |a| try o.own(a);
    const q = parts[0];
    return .{ .lin = .{ .w = fused.w, .s = fused.s, .b = fused.b, .bits = q.bits, .group_size = q.group_size, .mode = q.mode }, .widths = fused.widths };
}

/// The model-wide values attention layers share: the value-norm ones per head width, the proportional rope table.
pub const AttnShared = struct {
    v_norm: Array = .{},
    v_norm_global: Array = .{},
    freqs_global: Array = .{},

    pub fn build(o: *Owned, c: *const Config, s: mlx.mlx_stream) !AttnShared {
        var a: AttnShared = .{};
        if (c.has_v_norm) {
            a.v_norm = try ones(o, c.head_dim, s);
            if (c.dualHeadDim()) a.v_norm_global = try ones(o, c.global_head_dim, s);
        }
        if (c.rope_proportional) {
            a.freqs_global = try proportionalFreqs(c, s);
            try o.own(a.freqs_global);
        }
        return a;
    }
};

/// Gemma 4's proportional rope denominators for full layers: `factor * base^(arange(0, rotated, 2) / ghd)`, padded
/// with inf past the rotated dims. copied_from: engines/mlx/load/init.zig (`rope_freqs_global`).
fn proportionalFreqs(c: *const Config, s: mlx.mlx_stream) !Array {
    const ghd: u32 = if (c.global_head_dim > 0) c.global_head_dim else c.head_dim;
    const rotated: u32 = @intFromFloat(@as(f32, @floatFromInt(ghd)) * c.partial_rotary_factor_global);
    const n_rot: u32 = rotated / 2;
    const n_pad: u32 = (ghd - rotated) / 2;
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(rotated), 2, .float32, s));
    const ghd_s = mlx.mlx_array_new_float(@floatFromInt(ghd));
    defer _ = mlx.mlx_array_free(ghd_s);
    var ex = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ex);
    try mlx.check(mlx.mlx_divide(&ex, ar, ghd_s, s));
    const base = mlx.mlx_array_new_float(c.rope_theta);
    defer _ = mlx.mlx_array_free(base);
    var pow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pow);
    try mlx.check(mlx.mlx_power(&pow, base, ex, s));
    if (c.rope_proportional_factor != 1.0) {
        const f = mlx.mlx_array_new_float(c.rope_proportional_factor);
        defer _ = mlx.mlx_array_free(f);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, pow, f, s));
        _ = mlx.mlx_array_free(pow);
        pow = scaled;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (n_pad > 0) {
        const inf = mlx.mlx_array_new_float(std.math.inf(f32));
        defer _ = mlx.mlx_array_free(inf);
        var pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pad);
        try mlx.check(mlx.mlx_full(&pad, &[_]c_int{@intCast(n_pad)}, 1, inf, .float32, s));
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, pow);
        _ = mlx.mlx_vector_array_append_value(vec, pad);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    } else {
        try mlx.check(mlx.mlx_reshape(&out, pow, &[_]c_int{@intCast(n_rot + n_pad)}, 1, s));
    }
    return out;
}

/// Layer `li`'s head geometry and rope, as the engine's dense forward resolves it.
pub fn geoOf(c: *const Config, sh: *const AttnShared, li: u32, scale: f32) Geo {
    const global = c.isGlobalLayer(li);
    const dual = c.dualHeadDim() and global;
    const hd: u32 = if (dual) c.global_head_dim else c.head_dim;
    const gkv: u32 = if (c.num_global_key_value_heads > 0) c.num_global_key_value_heads else c.num_key_value_heads;
    const prop = global and sh.freqs_global.ctx != null;
    const partial: c_int = if (global and c.partial_rotary_factor_global < 1.0)
        @intCast(@as(u32, @intFromFloat(@as(f32, @floatFromInt(hd)) * c.partial_rotary_factor_global)))
    else
        @intCast(hd);
    return .{
        .heads = @intCast(c.num_attention_heads),
        .kv_heads = @intCast(if (dual) gkv else c.num_key_value_heads),
        .head_dim = @intCast(hd),
        .scale = scale,
        .rope_dims = if (prop) @intCast(hd) else partial,
        .rope_base = if (global) c.rope_theta else c.rope_local_base_freq,
        .rope_scale = if (prop) 1.0 else if (global) 1.0 / c.rope_scaling_factor else 1.0,
        .freqs = if (prop) sh.freqs_global else .{},
    };
}

/// Layer `li`'s attention: q/k/v/o through the format (a KV-shared layer projects only q), q|k|v joined where
/// quantized and its own, the per-head q/k norms. `angles` is the model's angle cache.
pub fn attentionAt(r: *Reader, o: *Owned, c: *const Config, sh: *const AttnShared, li: u32, scale: f32, angles: *qk_rope.AngleCache) !Attention.Weights {
    var nb: [256]u8 = undefined;
    const geo = geoOf(c, sh, li, scale);
    const hidden = c.hidden_size;
    const src = c.kvSource(li);
    const global = c.isGlobalLayer(li);
    const k_eq_v = src == null and c.attention_k_eq_v and global;
    var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{li}), hidden);
    var k: Linear = .{};
    var v: Linear = .{};
    if (src == null) {
        k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{li}), hidden);
        if (!k_eq_v) v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{li}), hidden);
    }
    const o_proj = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{li}), @intCast(geo.heads * geo.head_dim));
    const qkv = if (q.s.ctx != null and src == null and !k_eq_v) try join(o, &.{ &q, &k, &v }, r.s) else Joined{};
    for ([_]Linear{ q, k, v, o_proj }) |l| try o.linear(l);
    const q_norm = try r.norm(o, try r.name(&nb, "layers.{d}.self_attn.q_norm.weight", .{li}), c.norm_has_offset);
    const k_norm = if (src == null) try r.norm(o, try r.name(&nb, "layers.{d}.self_attn.k_norm.weight", .{li}), c.norm_has_offset) else Array{};
    const dual = c.dualHeadDim() and global;
    return .{
        .q = q,
        .k = k,
        .v = if (k_eq_v) k else v,
        .o = o_proj,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .qkv = qkv,
        .geo = geo,
        .window = if (c.has_sliding_window and !global) c.sliding_window else 0,
        .kv_source = src,
        .fit_causal = c.num_experts > 0,
        .v_norm = if (!c.has_v_norm) .{} else if (dual and sh.v_norm_global.ctx != null) sh.v_norm_global else sh.v_norm,
        .k_eq_v = k_eq_v,
        .angles = angles,
    };
}

/// The gated MLP's three projections, gate|up joined where quantized.
pub fn mlpAt(r: *Reader, o: *Owned, c: *const Config, li: u32, rows_joined: bool, compiled: anytype) !b.extras.GeGlu.Weights {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    // A double-wide layer's down projection reads its own width: the checkpoint's gate rows.
    const gate_w = r.src.tensor(try r.name(&nb, "layers.{d}.mlp.gate_proj.weight", .{li}));
    if (gate_w.ctx == null) return error.MissingWeight;
    const inter: u32 = @intCast(mlx.getShape(.{ .ctx = gate_w.ctx })[0]);
    var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{li}), hidden);
    var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{li}), hidden);
    const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{li}), inter);
    const gu = if (gate.s.ctx != null) try join(o, &.{ &gate, &up }, r.s) else Joined{};
    for ([_]Linear{ gate, up, down }) |l| try o.linear(l);
    return .{
        .gate = gate,
        .up = up,
        .down = down,
        .gu = gu,
        .rows_joined = rows_joined,
        .table = try compiled.geluTable(.bfloat16),
        .geglu = compiled.geglu,
        .gelu = compiled.gelu,
    };
}

/// The head as the engine finds it: under the decoder's own prefix, then at the root, then the tied embedding.
pub fn headOf(r: *Reader, o: *Owned, embed: Linear, hidden: u32, tied: bool) !Linear {
    var nb: [160]u8 = undefined;
    const nested = if (std.mem.eql(u8, r.prefix, "language_model.model")) "language_model" else r.prefix;
    for ([_][]const u8{ try std.fmt.bufPrint(&nb, "{s}.lm_head", .{nested}), "lm_head" }) |base| {
        var kb: [176]u8 = undefined;
        if (r.src.tensor(try std.fmt.bufPrint(&kb, "{s}.weight", .{base})).ctx == null) continue;
        const head = try r.tableAt(base, hidden);
        try o.linear(head);
        return head;
    }
    if (!tied) return error.MissingWeight;
    return embed;
}

/// The embedding table and head a Gemma model lends the engine.
pub fn TrunkWeight(comptime L: type) type {
    return struct {
        pub fn get(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
            const m: *L = @ptrCast(@alignCast(mp));
            const w = &m.base.weights;
            const l = switch (which) {
                .embed => w.embed.table,
                .head => w.head.head,
            };
            out.* = .{
                .w = .{ .ctx = l.w.ctx },
                .scales = .{ .ctx = l.s.ctx },
                .biases = .{ .ctx = l.b.ctx },
                .bits = @intCast(l.bits),
                .group_size = l.group_size,
                .layout = @backingInt(if (l.s.ctx == null) native.Layout.dense else switch (l.mode) {
                    .nvfp4 => native.Layout.nvfp4,
                    .mxfp4 => .mxfp4,
                    .mxfp8 => .mxfp8,
                    else => .affine,
                }),
            };
            return .ok;
        }
    };
}
