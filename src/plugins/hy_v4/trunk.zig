//! hy_v4_flash's trunk: the loaded per-layer weights (built once at load from the pack's map) and the text
//! forward — embed rows into `hc_mult` identical streams, each layer's [hc mix -> MLA -> hc write -> hc mix ->
//! dense/MoE FFN -> hc write], then the head collapse, the shared norm and the head.
//!
//! The forward follows `HYV4FlashTextModel`: `hidden = embed(ids).expand(hc_mult)`; a sublayer runs on
//! `norm(mixed)` and writes back `streams + post * update`; `norm(hc_head(streams))` feeds the head.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const abi = @import("../../abi/plugin.zig");
const block = @import("../../sdk/bases/block.zig");
const embedding = @import("../../sdk/blocks/embedding.zig");
const proj = @import("blocks/proj.zig");
const ihc = @import("blocks/ihc.zig");
const mla = @import("blocks/mla.zig");
const moe = @import("blocks/moe.zig");
const config = @import("config.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const log = @import("../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;
const C = config.Config;
const gpa = std.heap.c_allocator;

pub const Layer = struct {
    input_norm: Array,
    post_attn_norm: Array,
    attn: mla.Attn,
    hc_attn: ihc.Params,
    hc_mlp: ihc.Params,
    ffn: union(enum) { dense: moe.Dense, sparse: moe.Moe },
};

pub const Model = struct {
    c: *const C,
    layers: []Layer,
    embed: Proj,
    lm_head: Proj,
    /// `model.norm.weight`, the shared norm after the head collapse.
    norm: Array,
    head: ihc.HeadParams,
    /// The table's declared caps (what the forward serves).
    caps: abi.Caps,
    shared: block.Shared,
    /// Derived arrays built at load (transposes, dense halves): freed in `deinit`.
    owned: std.ArrayList(Array),

    pub fn deinit(m: *Model) void {
        for (m.owned.items) |x| _ = mlx.mlx_array_free(x);
        m.owned.deinit(gpa);
        gpa.free(m.layers);
        m.shared.deinit();
        gpa.destroy(m);
    }

    /// The text forward: `f.tokens`/`f.ids` in, logits `[rows, vocab]` out.
    pub fn forward(m: *Model, f: *const abi.Forward) !abi.ForwardOut {
        const c = m.c;
        const slots: c_int = @intCast(if (f.slots.len > 0) f.slots.len else 1);
        const rows: c_int = @intCast(f.rows);
        const hidden: c_int = @intCast(c.hidden_size);
        const hc: c_int = @intCast(c.hc_mult);
        const vocab: c_int = @intCast(c.vocab_size);

        var g = Graph.init(gpa, mlx.gpuStream());
        defer g.deinit();

        const ids = if (f.ids.ctx != null)
            try g.adopt(.{ .ctx = f.ids.ctx })
        else blk: {
            const buf = try gpa.alloc(i32, f.tokens.len);
            defer gpa.free(buf);
            for (f.tokens.slice(), 0..) |tok, i| buf[i] = @intCast(tok);
            break :blk try g.fromSlice(i32, buf, &.{ slots, rows });
        };

        const x = try m.embedRows(&g, ids); // [slots, rows, hidden]
        var streams = try g.broadcast(try g.expandDims(x, -2), &.{ slots, rows, hc, hidden });

        var ctx: block.Ctx = .{
            .g = &g,
            .mode = f.mode,
            .rows = rows,
            .slots = slots,
            .positions = if (f.positions.len > 0) f.positions.slice() else &.{0},
            .caches = f.caches.slice(),
            .cache_ops = f.cache_ops,
            .shared = &m.shared,
        };

        for (m.layers, 0..) |*l, li| {
            ctx.layer = @intCast(li);
            const pa = try ihc.pre(&g, streams, &l.hc_attn);
            defer {
                _ = mlx.mlx_array_free(pa.post);
                _ = mlx.mlx_array_free(pa.mixed);
            }
            const normed = try g.rmsNorm(pa.mixed, l.input_norm, c.rms_norm_eps);
            const y = try l.attn.forwards(&ctx, normed);
            streams = try g.adopt(try ihc.combine(&g, streams, y, pa.post));

            const pm = try ihc.pre(&g, streams, &l.hc_mlp);
            defer {
                _ = mlx.mlx_array_free(pm.post);
                _ = mlx.mlx_array_free(pm.mixed);
            }
            const normed_m = try g.rmsNorm(pm.mixed, l.post_attn_norm, c.rms_norm_eps);
            const z = switch (l.ffn) {
                .dense => |*d| try d.forward(&g, normed_m),
                .sparse => |*mm| try mm.forward(&g, normed_m),
            };
            streams = try g.adopt(try ihc.combine(&g, streams, z, pm.post));

            if (f.eval_every > 0 and (li + 1) % f.eval_every == 0) try ops.eval(streams);
        }

        const collapsed = try g.adopt(try ihc.head(&g, streams, &m.head)); // [slots, rows, hidden]
        const normed_f = try g.rmsNorm(collapsed, m.norm, c.rms_norm_eps);
        const logits = try m.lm_head.forward(&g, normed_f); // [slots, rows, vocab]

        var out: abi.ForwardOut = .{};
        switch (f.logits) {
            .last_row => {
                const last = try g.slice(logits, &.{ 0, rows - 1, 0 }, &.{ slots, rows, vocab });
                // The host owns what `ForwardOut` carries: escape it, never leave it in the graph.
                out.logits = .{ .ctx = g.escape(try g.reshape(last, &.{ slots, vocab })).ctx };
            },
            .all_rows => out.logits = .{ .ctx = g.escape(try g.reshape(logits, &.{ slots * rows, vocab })).ctx },
            .none => {},
        }
        if (f.want_hidden) out.hidden = .{ .ctx = g.escape(collapsed).ctx };
        return out;
    }

    /// Embedding rows for `ids` `[slots, rows]`: a quantized table gathers its rows (never dequantized whole), a
    /// dense one takes them (both stored `[vocab, in]`).
    fn embedRows(m: *Model, g: *Graph, ids: Array) !Array {
        if (m.embed.s.ctx != null) {
            const rows = embedding.gatherQuantizedRows(g.s, m.embed.w, m.embed.s, m.embed.b, ids,
                m.embed.gs, m.embed.bits, .affine, .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }) catch |e| {
                log.err("hy_v4: embed gather: {t}\n", .{e});
                return e;
            };
            return g.adopt(rows);
        }
        return g.take(m.embed.w, ids, 0);
    }
};

/// Builds the trunk from the loaded map: every layer's projections, hyper-connections and FFN.
pub fn build(c: *const C, w: *const weights_load.Weights) !*Model {
    var owned_store: std.ArrayList(Array) = .empty;
    const owned = &owned_store;
    errdefer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(gpa);
    }
    const layers = try gpa.alloc(Layer, c.num_hidden_layers);
    errdefer gpa.free(layers);

    const head_fn = try transposeOwned(owned, try require(w, "model.hc_head.hc_head_fn", .{}));
    for (layers, 0..) |*l, li| {
        l.input_norm = try require(w, "model.layers.{d}.input_layernorm.weight", .{@as(u32, @intCast(li))});
        l.post_attn_norm = try require(w, "model.layers.{d}.post_attention_layernorm.weight", .{@as(u32, @intCast(li))});
        l.attn = try buildAttn(c, w, owned, @intCast(li));
        l.hc_attn = try buildHc(c, w, owned, "model.layers.{d}.hc_attn_layer.hc_pre", @intCast(li));
        l.hc_mlp = try buildHc(c, w, owned, "model.layers.{d}.hc_mlp_layer.hc_pre", @intCast(li));
        if (li < c.first_k_dense_replace) {
            l.ffn = .{ .dense = .{
                .gate = try buildBank(c, w, owned, "model.layers.{d}.mlp.gate_proj", @intCast(li)),
                .up = try buildBank(c, w, owned, "model.layers.{d}.mlp.up_proj", @intCast(li)),
                .down = try buildBank(c, w, owned, "model.layers.{d}.mlp.down_proj", @intCast(li)),
                .post_fc2_norm = try require(w, "model.layers.{d}.mlp.post_fc2_norm.weight", .{@as(u32, @intCast(li))}),
                .rms_eps = c.rms_norm_eps,
            } };
        } else {
            l.ffn = .{ .sparse = .{
                .router = .{
                    .gate_t = try transposeOwned(owned, try require(w, "model.layers.{d}.mlp.gate.weight", .{@as(u32, @intCast(li))})),
                    .bias = try require(w, "model.layers.{d}.mlp.gate.e_score_correction_bias", .{@as(u32, @intCast(li))}),
                    .experts = c.n_routed_experts,
                    .hidden = c.hidden_size,
                    .top_k = c.num_experts_per_tok,
                    .norm = true,
                    .scale = c.routed_scaling_factor,
                    .n_group = c.n_group,
                    .topk_group = c.topk_group,
                },
                .routed_down = try buildBank(c, w, owned, "model.layers.{d}.mlp.routed_expert_down_proj", @intCast(li)),
                .routed_up = try buildBank(c, w, owned, "model.layers.{d}.mlp.routed_expert_up_proj", @intCast(li)),
                .gate_up = try buildExperts(c, w, "model.layers.{d}.mlp.experts.gate_up_proj", @intCast(li)),
                .down = try buildExperts(c, w, "model.layers.{d}.mlp.experts.down_proj", @intCast(li)),
                .shared_gate = try buildBank(c, w, owned, "model.layers.{d}.mlp.shared_experts.gate_proj", @intCast(li)),
                .shared_up = try buildBank(c, w, owned, "model.layers.{d}.mlp.shared_experts.up_proj", @intCast(li)),
                .shared_down = try buildBank(c, w, owned, "model.layers.{d}.mlp.shared_experts.down_proj", @intCast(li)),
                .combine_output_norm = try require(w, "model.layers.{d}.mlp.combine_output_norm.weight", .{@as(u32, @intCast(li))}),
                .rms_eps = c.rms_norm_eps,
                .swiglu_limit = c.swiglu_limit,
            } };
        }
    }

    const m = try gpa.create(Model);
    m.* = .{
        .c = c,
        .layers = layers,
        .embed = try buildBank(c, w, owned, "model.embed_tokens", null),
        .lm_head = try buildBank(c, w, owned, "lm_head", null),
        .norm = try require(w, "model.norm.weight", .{}),
        .head = .{
            .fn_t = head_fn,
            .base = try require(w, "model.hc_head.hc_head_base", .{}),
            .scale = try require(w, "model.hc_head.hc_head_scale", .{}),
            .lanes = c.hc_mult,
            .hidden = c.hidden_size,
            .eps = c.hc_eps,
            .rms_eps = c.rms_norm_eps,
        },
        .caps = .{ .moe = c.n_routed_experts > 0, .images = c.has_vision, .batches_decode = false, .batch_pads_kv = false },
        .shared = block.Shared.init(c.rms_norm_eps, c.rms_norm_eps),
        .owned = owned.*,
    };
    return m;
}

fn buildAttn(c: *const C, w: *const weights_load.Weights, owned: *std.ArrayList(Array), li: u32) !mla.Attn {
    const heads = config.layerHeads(c, li);
    const kv_w = try require(w, "model.layers.{d}.self_attn.kv_b_proj.weight", .{li});
    const kv_s = live(w.get(try fmtName(gpa, "model.layers.{d}.self_attn.kv_b_proj.scales", li)));
    const kv_b = live(w.get(try fmtName(gpa, "model.layers.{d}.self_attn.kv_b_proj.biases", li)));
    const halves = try kvHalves(c, kv_w, kv_s, kv_b, heads);
    try owned.append(gpa, halves[0]);
    try owned.append(gpa, halves[1]);
    return .{
        .q_a = try buildBank(c, w, owned, "model.layers.{d}.self_attn.q_a_proj", li),
        .q_a_norm = try require(w, "model.layers.{d}.self_attn.q_a_layernorm.weight", .{li}),
        .q_b = try buildBank(c, w, owned, "model.layers.{d}.self_attn.q_b_proj", li),
        .kv_a = try buildBank(c, w, owned, "model.layers.{d}.self_attn.kv_a_proj_with_mqa", li),
        .kv_a_norm = try require(w, "model.layers.{d}.self_attn.kv_a_layernorm.weight", .{li}),
        .wk = halves[0],
        .wv = halves[1],
        .linear_gate = try buildBank(c, w, owned, "model.layers.{d}.self_attn.linear_gate", li),
        .o_proj = try buildBank(c, w, owned, "model.layers.{d}.self_attn.o_proj", li),
        .post_proj_norm = try require(w, "model.layers.{d}.self_attn.post_proj_norm.weight", .{li}),
        .sink = try require(w, "model.layers.{d}.self_attn.learnable_sink_param", .{li}),
        .heads = heads,
        .nope = c.qk_nope_head_dim,
        .rope_dim = c.qk_rope_head_dim,
        .v_dim = c.v_head_dim,
        .rank = c.kv_lora_rank,
        .theta = config.layerTheta(c, li),
        .window = if (li < c.dsa_layers.len and c.dsa_layers[li]) 0 else c.sliding_window,
        .rms_eps = c.rms_norm_eps,
    };
}

fn buildHc(c: *const C, w: *const weights_load.Weights, owned: *std.ArrayList(Array), comptime fmt: []const u8, li: u32) !ihc.Params {
    return .{
        .fn_t = try transposeOwned(owned, try fmtFmt(w, fmt ++ ".hc_fn", li)),
        .base = try fmtFmt(w, fmt ++ ".hc_base", li),
        .scale = try fmtFmt(w, fmt ++ ".hc_scale", li),
        .lanes = c.hc_mult,
        .hidden = c.hidden_size,
        .magnitude = c.hc_magnitude,
        .eps = c.hc_eps,
        .rms_eps = c.rms_norm_eps,
    };
}

/// One projection as the pack stores it: packed affine `[out, in/8]` with bf16 leaves, or a dense arm the SDK
/// loader rewrote (transposed to `[in, out]` once, since `Proj.forward` reads dense weights that way).
fn buildBank(c: *const C, w: *const weights_load.Weights, owned: *std.ArrayList(Array), comptime fmt: []const u8, li: ?u32) !Proj {
    const wn = try bankName(gpa, fmt, ".weight", li);
    errdefer gpa.free(wn);
    const weight = w.get(wn) orelse return error.MissingWeight;
    defer gpa.free(wn);
    if (li) |i| {
        const sn = try bankName(gpa, fmt, ".scales", i);
        defer gpa.free(sn);
        if (live(w.get(sn))) |s| {
            const bn = try bankName(gpa, fmt, ".biases", i);
            defer gpa.free(bn);
            return .{ .w = weight, .s = s, .b = live(w.get(bn)) orelse .{}, .bits = c.quant_bits, .gs = c.quant_group_size };
        }
    } else {
        const sn = try bankName(gpa, fmt, ".scales", null);
        defer gpa.free(sn);
        if (live(w.get(sn))) |s| {
            const bn = try bankName(gpa, fmt, ".biases", null);
            defer gpa.free(bn);
            return .{ .w = weight, .s = s, .b = live(w.get(bn)) orelse .{}, .bits = c.quant_bits, .gs = c.quant_group_size };
        }
    }
    // Dense (rewritten): readable only through a transposed copy, which the load must own. `build` appends it.
    return .{ .w = try transposeOwned(owned, weight) };
}

fn buildExperts(c: *const C, w: *const weights_load.Weights, comptime fmt: []const u8, li: u32) !proj.ExpertBank {
    const wn = try bankName(gpa, fmt, ".weight", li);
    defer gpa.free(wn);
    const weight = w.get(wn) orelse return error.MissingWeight;
    const sn = try bankName(gpa, fmt, ".scales", li);
    defer gpa.free(sn);
    const bn = try bankName(gpa, fmt, ".biases", li);
    defer gpa.free(bn);
    return .{ .w = weight, .s = live(w.get(sn)) orelse .{}, .b = live(w.get(bn)) orelse .{}, .bits = c.quant_bits, .gs = c.quant_group_size };
}

/// `kv_b_proj` split into its K half `[heads, nope, rank]` and V half `[heads, v_dim, rank]`, dense.
fn kvHalves(c: *const C, weight: Array, scales: ?Array, biases: ?Array, heads: u32) ![2]Array {
    const rank: c_int = @intCast(c.kv_lora_rank);
    const nope: c_int = @intCast(c.qk_nope_head_dim);
    const vd: c_int = @intCast(c.v_head_dim);
    const h: c_int = @intCast(heads);
    var dense: Array = .{};
    var owned_dense = false;
    if (scales) |s| {
        dense = mlx.mlx_array_new();
        owned_dense = true;
        errdefer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_dequantize(&dense, weight, s, biases orelse .{},
            mlx.mlx_optional_int.some(@intCast(c.quant_group_size)),
            mlx.mlx_optional_int.some(@intCast(c.quant_bits)), "affine",
            .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(dense));
    } else {
        dense = weight;
    }

    const shaped = try reshapeOwned(dense, &.{ h, nope + vd, rank });
    defer _ = mlx.mlx_array_free(shaped);
    var out: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out[0]);
        _ = mlx.mlx_array_free(out[1]);
    }
    try mlx.check(mlx.mlx_slice(&out[0], shaped, &[_]c_int{ 0, 0, 0 }, 3,
        &[_]c_int{ h, nope, rank }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()));
    try mlx.check(mlx.mlx_slice(&out[1], shaped, &[_]c_int{ 0, nope, 0 }, 3,
        &[_]c_int{ h, nope + vd, rank }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(out[0]));
    try mlx.check(mlx.mlx_array_eval(out[1]));
    if (owned_dense) _ = mlx.mlx_array_free(dense);
    return out;
}

fn transposeOwned(owned: *std.ArrayList(Array), src: Array) !Array {
    const t = try transposeTo(src);
    try owned.append(gpa, t);
    return t;
}

fn transposeTo(x: Array) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_transpose_axes(&out, x, &[_]c_int{ 1, 0 }, 2, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn reshapeOwned(x: Array, sh: []const c_int) !Array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, x, sh.ptr, sh.len, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A map handle that still holds an array: the SDK loader leaves a freed quant leaf as an empty handle.
fn live(x: ?Array) ?Array {
    return if (x) |a| if (a.ctx != null) a else null else null;
}

/// A tensor name at an optional layer index.
fn bankName(a: std.mem.Allocator, comptime fmt: []const u8, comptime suffix: []const u8, li: ?u32) ![]const u8 {
    if (comptime std.mem.indexOf(u8, fmt, "{d}") != null) return std.fmt.allocPrint(a, fmt ++ suffix, .{li.?});
    return std.fmt.allocPrint(a, fmt ++ suffix, .{});
}

/// A tensor name at a layer index (no suffix).
fn fmtName(a: std.mem.Allocator, comptime fmt: []const u8, li: u32) ![]const u8 {
    return std.fmt.allocPrint(a, fmt, .{li});
}

/// A packed name with a layer index (the caller's `fmt` may carry `{d}`).
fn fmtFmt(w: *const weights_load.Weights, comptime fmt: []const u8, li: u32) !Array {
    const name = try std.fmt.allocPrint(gpa, fmt, .{li});
    defer gpa.free(name);
    return w.get(name) orelse error.MissingWeight;
}

fn require(w: *const weights_load.Weights, comptime fmt: []const u8, args: anytype) !Array {
    const name = try std.fmt.allocPrint(gpa, fmt, args);
    defer gpa.free(name);
    return w.get(name) orelse error.MissingWeight;
}
