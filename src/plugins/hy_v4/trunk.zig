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
const indexer_mod = @import("blocks/indexer.zig");
const engram_mod = @import("blocks/engram.zig");
const numpy_rng = @import("blocks/numpy_rng.zig");
const vision_mod = @import("vision.zig");
const dfly_mod = @import("blocks/dfly.zig");
const sliding = @import("../../sdk/bases/blocks/sliding.zig");
const Verdict = abi.Verdict;
const TargetLayers = dfly_mod.TargetLayers;
const config = @import("config.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const weights = @import("weights.zig");
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
    /// The Engram memory injected before attention (null = not an Engram layer).
    engram: ?*engram_mod.Engram = null,
    ffn: union(enum) { dense: moe.Dense, sparse: moe.Moe },
};

pub const SlotState = struct {
    /// The last row's target hidden (post final norm), `[1, hidden]`.
    hidden: Array = .{},
    /// A verify's rows' hidden `[rows, hidden]`, for the accepted row's rollback.
    verify_hidden: Array = .{},
    verify_tokens: [abi.max_round_tokens]u32 = @splat(0),
    verify_n: u32 = 0,
    verify_open: bool = false,
    cache: abi.Handle = null,
    cache_ops: ?*const abi.CacheOps = null,
    anchor: u64 = 0,
    active: bool = false,
};

pub const Model = struct {
    c: *const C,
    layers: []Layer,
    embed: Proj,
    lm_head: Proj,
    /// `model.norm.weight`, the shared norm after the head collapse.
    norm: Array,
    head: ihc.HeadParams,
    /// The SigLIP-style tower (null = no vision weights in the map / --no-vision).
    vision: ?*vision_mod.Vision = null,
    /// The DFly2 drafter (null = no draft weights).
    dfly: ?*dfly_mod.Drafter = null,
    /// Per-slot drafter state: the last target hidden (post final norm), the cache handle and an open verify.
    slot_state: [64]SlotState = @splat(.{}),
    /// The slot the most recent forward ran on (`slotCacheHandle`/`slotCacheOps`).
    active_slot: u32 = 0,
    /// The table's declared caps (what the forward serves).
    caps: abi.Caps,
    shared: block.Shared,
    /// Derived arrays built at load (transposes, dense halves): freed in `deinit`.
    owned: std.ArrayList(Array),

    pub fn deinit(m: *Model) void {
        if (m.vision) |v| v.deinit();
        if (m.dfly) |df| df.deinit();
        for (&m.slot_state) |*st| {
            if (st.hidden.ctx != null) _ = mlx.mlx_array_free(st.hidden);
            if (st.verify_hidden.ctx != null) _ = mlx.mlx_array_free(st.verify_hidden);
        }
        for (m.layers) |*l| if (l.engram) |en| {
            gpa.free(en.lookup_host);
            gpa.destroy(en);
        };
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

        // The host owns the id array: borrow it, never adopt it into this graph (its deinit would free the
        // host's handle twice).
        const ids: Array = if (f.ids.ctx != null)
            .{ .ctx = f.ids.ctx }
        else blk: {
            const buf = try gpa.alloc(i32, f.tokens.len);
            defer gpa.free(buf);
            for (f.tokens.slice(), 0..) |tok, i| buf[i] = @intCast(tok);
            break :blk try g.fromSlice(i32, buf, &.{ slots, rows });
        };

        const x = try m.embedRows(&g, ids); // [slots, rows, hidden]
        // Splice the vision tower's rows at the image placeholders, from `media_offset` on (a chunked
        // prefill resumes there): every placeholder row takes the next tower row; the rest keep their
        // token embedding.
        var spliced = x;
        if (f.media.ctx != null and slots == 1) {
            const media: Array = .{ .ctx = f.media.ctx };
            const msh = ops.shape(media);
            if (msh.len == 3) {
                try ops.eval(ids);
                const tok = mlx.mlx_array_data_int32(ids) orelse return error.NoData;
                const total: usize = @as(usize, @intCast(rows)) * @as(usize, @intCast(slots));
                const idx = try gpa.alloc(i32, total);
                defer gpa.free(idx);
                var k: usize = f.media_offset;
                for (0..total) |i| {
                    if (tok[i] == @as(i32, @intCast(c.image_token_id))) {
                        idx[i] = @intCast(k);
                        k += 1;
                    } else idx[i] = 0;
                }
                const idx_arr = try g.fromSlice(i32, idx, &.{ slots, rows });
                const flat_media = try g.reshape(media, &.{ msh[1], msh[2] });
                const picked = try g.take(flat_media, idx_arr, 0); // [slots, rows, hidden]
                const mask = try g.expandDims(try g.equal(ids, try g.scalarLike(ids, @as(f32, @floatFromInt(c.image_token_id)))), -1);
                spliced = try g.where(mask, picked, x);
            }
        }
        var streams = try g.broadcast(try g.expandDims(spliced, -2), &.{ slots, rows, hc, hidden });

        var ctx: block.Ctx = .{
            .g = &g,
            .mode = f.mode,
            .rows = rows,
            .slots = slots,
            .positions = if (f.positions.len > 0) f.positions.slice() else &.{0},
            .mrope_positions = if (f.mrope_positions.ctx != null) .{ .ctx = f.mrope_positions.ctx } else null,
            .caches = f.caches.slice(),
            .cache_ops = f.cache_ops,
            .shared = &m.shared,
        };

        for (m.layers, 0..) |*l, li| {
            ctx.layer = @intCast(li);
            // The Engram residual lands on the streams BEFORE the attention mix, as the reference injects it.
            if (l.engram) |en| {
                const r = try en.forward(&g, streams, ids);
                streams = try g.add(streams, r);
            }
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
        // The drafter fuses the last accepted token's POST-norm hidden (HF returns `norm(self.hc_head(...))`).
        const hidden_rows = try g.reshape(normed_f, &.{ slots, rows, hidden });
        if (f.want_hidden) out.hidden = .{ .ctx = g.escape(hidden_rows).ctx };
        // Keep the slot state the native drafter reads: hidden last row, the verify rows, and the cache.
        {
            const sid: usize = if (f.slots.len > 0) f.slots.slice()[0] else 0;
            if (sid < m.slot_state.len) {
                const st = &m.slot_state[sid];
                if (st.hidden.ctx != null) _ = mlx.mlx_array_free(st.hidden);
                if (st.verify_hidden.ctx != null) _ = mlx.mlx_array_free(st.verify_hidden);
                st.hidden = .{};
                st.verify_hidden = .{};
                st.cache = if (f.caches.len > 0) f.caches.slice()[0] else null;
                st.cache_ops = f.cache_ops;
                st.anchor = if (f.positions.len > 0) @as(u64, @intCast(f.positions.slice()[0])) + @as(u64, @intCast(rows)) else @as(u64, @intCast(rows));
                st.active = true;
                if (f.logits == .all_rows) {
                    st.verify_hidden = .{ .ctx = g.escape(try g.reshape(normed_f, &.{ slots * rows, hidden })).ctx };
                }
                const lr = try g.slice(normed_f, &.{ 0, rows - 1, 0 }, &.{ slots, rows, hidden });
                st.hidden = .{ .ctx = g.escape(try g.reshape(lr, &.{ slots, hidden })).ctx };
                m.active_slot = @intCast(sid);
            }
        }
        return out;
    }

    /// Embedding rows for `ids` `[slots, rows]`: a quantized table gathers its rows (never dequantized whole), a
    /// dense one takes them (both stored `[vocab, in]`).
    pub fn embedRows(m: *Model, g: *Graph, ids: Array) !Array {
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
    /// One row's logits through the head.
    pub fn lm_head_row(m: *Model, g: *Graph, x: Array) !Array {
        return m.lm_head.forward(g, x);
    }

    pub fn slotCacheHandle(m: *const Model) ?abi.Handle {
        if (m.active_slot >= m.slot_state.len) return null;
        return m.slot_state[m.active_slot].cache;
    }

    pub fn slotCacheOps(m: *const Model) ?*const abi.CacheOps {
        if (m.active_slot >= m.slot_state.len) return null;
        return m.slot_state[m.active_slot].cache_ops;
    }
};

/// Builds the trunk from the loaded map: every layer's projections, hyper-connections and FFN.
pub fn build(c: *const C, w: *const weights_load.Weights, tables: []const weights.Engram) !*Model {
    var owned_store: std.ArrayList(Array) = .empty;
    const owned = &owned_store;
    errdefer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(gpa);
    }
    const layers = try gpa.alloc(Layer, c.num_hidden_layers);
    errdefer gpa.free(layers);

    const head_fn = try transposeOwned(owned, try require(w, "model.hc_head.hc_head_fn", .{}));
    var engram_i: usize = 0;
    for (layers, 0..) |*l, li| {
        l.engram = null; // `gpa.alloc` leaves the slice undefined; only Engram layers fill it
        l.input_norm = try require(w, "model.layers.{d}.input_layernorm.weight", .{@as(u32, @intCast(li))});
        l.post_attn_norm = try require(w, "model.layers.{d}.post_attention_layernorm.weight", .{@as(u32, @intCast(li))});
        l.attn = try buildAttn(c, w, owned, @intCast(li));
        l.hc_attn = try buildHc(c, w, owned, "model.layers.{d}.hc_attn_layer.hc_pre", @intCast(li));
        if (li < c.engram_layers.len and c.engram_layers[li]) {
            if (engram_i >= tables.len) return error.EngramTableMissing;
            l.engram = try buildEngram(c, w, owned, tables[engram_i], @intCast(li));
            engram_i += 1;
        }
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
    errdefer m.deinit();
    if (c.has_vision and w.get("vit.embeddings.patch_embedding.weight") != null) {
        m.vision = try vision_mod.load(c, w);
    }
    if (c.dfly_layers > 0 and w.get("model.dfly_layers.final_norm.weight") != null) {
        m.dfly = try dfly_mod.load(c, w);
    }
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
        .mrope_sections = c.mrope_section,
        .window = if (li < c.dsa_layers.len and c.dsa_layers[li]) 0 else c.sliding_window,
        .indexer = if (li < c.dsa_layers.len and c.dsa_layers[li]) try buildIndexer(c, w, owned, li) else null,
        .rms_eps = c.rms_norm_eps,
    };
}

/// A layer's Engram memory: the frozen lookup table (host copy), numpy-exact multipliers, the per-head moduli and
/// the projections/norms/scales the residual math reads.
fn buildEngram(c: *const C, w: *const weights_load.Weights, owned: *std.ArrayList(Array), tbl: weights.Engram, li: u32) !*engram_mod.Engram {
    const e = try gpa.create(engram_mod.Engram);
    errdefer gpa.destroy(e);
    const lookup = try require(w, "model.layers.{d}.engram_layer.lookup_table", .{li});
    // Evaluate the table once and keep a host copy: the hash is host-side int arithmetic. The pack's table and
    // the config's `engram_lookup_table_size` must agree, or the multipliers were drawn over another range.
    try mlx.check(mlx.mlx_array_eval(lookup));
    const lsize = mlx.mlx_array_shape(lookup)[0];
    if (lsize != @as(c_int, @intCast(c.engram_lookup_size))) return error.LookupTableSizeMismatch;
    const src = mlx.mlx_array_data_int32(lookup) orelse return error.NoData;
    const host = try gpa.alloc(i32, c.engram_lookup_size);
    errdefer gpa.free(host);
    @memcpy(host, src[0..c.engram_lookup_size]);
    // The multiplier bound is drawn from the LOADED map's compressed vocabulary (`max+1`), not the table's
    // length: the pack's map is trained, so the two differ and a bound off its max reads the wrong rows.
    var max_comp: i32 = 0;
    for (host) |v| {
        if (v < 0) return error.LookupTableInvalid;
        max_comp = @max(max_comp, v);
    }
    const compressed_vocab: u64 = @as(u64, @intCast(max_comp)) + 1;
    // The multipliers are numpy's draw for this layer's HF id (1-based), not stored in the pack.
    var mult: [3]i64 = undefined;
    numpy_rng.hashMultipliers(c.engram_seed + 10007 * (li + 1), compressed_vocab, 3,
        @as([*]u64, @ptrCast(&mult))[0..3]);

    var moduli: [16]u32 = @splat(0);
    var offs: [16]u32 = @splat(0);
    var acc: u32 = 0;
    for (0..16) |h| {
        moduli[h] = c.engram_moduli[li][h];
        offs[h] = acc;
        acc += moduli[h];
    }
    e.* = .{
        .tbl = tbl,
        .lookup_host = host,
        .mult = mult,
        .moduli = moduli,
        .head_off = offs,
        // One head per (order, head slot) pair: orders 2..max_ngram, each `engram_n_head_per_ngram` wide.
        .heads = (c.engram_max_ngram - 1) * c.engram_ngram_heads,
        .head_dim = c.engram_head_dim,
        .emb_width = (c.engram_max_ngram - 1) * c.engram_ngram_heads * c.engram_head_dim,
        .key_proj = .{
            try buildBank(c, w, owned, "model.layers.{d}.engram_layer.key_projs.0", li),
            try buildBank(c, w, owned, "model.layers.{d}.engram_layer.key_projs.1", li),
            try buildBank(c, w, owned, "model.layers.{d}.engram_layer.key_projs.2", li),
            try buildBank(c, w, owned, "model.layers.{d}.engram_layer.key_projs.3", li),
        },
        .key_bias = .{
            try require(w, "model.layers.{d}.engram_layer.key_projs.0.bias", .{li}),
            try require(w, "model.layers.{d}.engram_layer.key_projs.1.bias", .{li}),
            try require(w, "model.layers.{d}.engram_layer.key_projs.2.bias", .{li}),
            try require(w, "model.layers.{d}.engram_layer.key_projs.3.bias", .{li}),
        },
        .norm_key = .{
            try require(w, "model.layers.{d}.engram_layer.norm_key.0.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_key.1.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_key.2.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_key.3.weight", .{li}),
        },
        .norm_query = .{
            try require(w, "model.layers.{d}.engram_layer.norm_query.0.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_query.1.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_query.2.weight", .{li}),
            try require(w, "model.layers.{d}.engram_layer.norm_query.3.weight", .{li}),
        },
        .value_proj = try buildBank(c, w, owned, "model.layers.{d}.engram_layer.value_proj", li),
        .value_bias = try require(w, "model.layers.{d}.engram_layer.value_proj.bias", .{li}),
        .head_norm_w = try stackHeadNorms(w, owned, li),
        .pre_gamma = try require(w, "model.layers.{d}.engram_layer.unified_pre_gamma", .{li}),
        .hc_scales = .{
            try require(w, "model.layers.{d}.engram_layer.unified_per_hc_scales.0", .{li}),
            try require(w, "model.layers.{d}.engram_layer.unified_per_hc_scales.1", .{li}),
            try require(w, "model.layers.{d}.engram_layer.unified_per_hc_scales.2", .{li}),
            try require(w, "model.layers.{d}.engram_layer.unified_per_hc_scales.3", .{li}),
        },
        .pad_id = c.engram_pad_id,
        .image_token_id = c.image_token_id,
        .video_token_id = c.video_token_id,
        .lookup_size = c.engram_lookup_size,
        .fp8 = c.engram_fp8,
        .hc_mult = c.hc_mult,
        .hidden = c.hidden_size,
    };
    return e;
}

/// The 16 per-head norm weights `[96]` stacked to `[16, 96]`, evaluated once at load.
fn stackHeadNorms(w: *const weights_load.Weights, owned: *std.ArrayList(Array), li: u32) !Array {
    var parts: [16]Array = undefined;
    for (0..16) |h| parts[h] = try require(w, "model.layers.{d}.engram_layer.lookup_per_head_norms.{d}.weight", .{ li, @as(u32, @intCast(h)) });
    var g = ops.Graph.init(gpa, mlx.gpuStream());
    defer g.deinit();
    const cat = try g.concat(&parts, -1);
    const shaped = try g.reshape(cat, &.{ 16, @intCast(@as(u32, 96)) });
    try mlx.check(mlx.mlx_array_eval(shaped));
    const escaped = g.escape(shaped);
    try owned.append(gpa, escaped);
    return escaped;
}

/// A DSA layer's compressed indexer: packed query/token-key/head-weight projections, the dense fp32 pooling
/// matrices, the norms and the intra-group position embedding.
fn buildIndexer(c: *const C, w: *const weights_load.Weights, owned: *std.ArrayList(Array), li: u32) !*const indexer_mod.Indexer {
    const ci = "model.layers.{d}.self_attn.compressed_indexer";
    const ix = try gpa.create(indexer_mod.Indexer);
    ix.* = .{
        .query = try buildBank(c, w, owned, ci ++ ".query_projection", li),
        .head_weight = try buildBank(c, w, owned, ci ++ ".head_weight_projection", li),
        .token_key = try buildBank(c, w, owned, ci ++ ".token_key_projection", li),
        .token_key_norm = try require(w, ci ++ ".token_key_norm.weight", .{li}),
        .value_proj = try denseProj(owned, try require(w, ci ++ ".key_value_projection.weight", .{li})),
        .gate_proj = try denseProj(owned, try require(w, ci ++ ".key_gate_projection.weight", .{li})),
        .key_norm = try require(w, ci ++ ".key_norm.weight", .{li}),
        .ape = try require(w, ci ++ ".intra_group_ape", .{li}),
        .heads = c.index_n_heads,
        .dim = c.index_head_dim,
        .ratio = c.compressed_ratio,
        .group_topk = c.compressed_group_topk,
        .budget = c.index_topk,
        .rope_dim = c.qk_rope_head_dim,
        .theta = config.layerTheta(c, li),
        .rms_eps = c.rms_norm_eps,
    };
    return ix;
}

/// A dense fp32 weight (`[out, in]` on disk) as a `Proj` reading `[in, out]`.
fn denseProj(owned: *std.ArrayList(Array), src: Array) !Proj {
    const t = try transposeTo(src);
    try owned.append(gpa, t);
    return .{ .w = t };
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

/// The block's draft tokens (the round's slots 1..): ids `[1,B]` (slot 0 is `t1`), `target_hidden` `[hidden]`.
pub fn dflyDraft(m: *Model, d: *dfly_mod.Drafter, ids: Array, positions: []const u32, target_hidden: Array) ![]u32 {
    const B: usize = @intCast(@as(c_int, @intCast(ops.shape(ids)[1])));
    const hidden: c_int = @intCast(d.hidden);
    var g = Graph.init(gpa, d.s);
    defer g.deinit();

    const embeds = try m.embedRows(&g, ids); // [1,B,hidden]
    var x = try g.reshape(embeds, &.{ @intCast(B), hidden });
    // mtp_fusion on slot 0 only: anchors + gate * eh_proj(cat(enorm(anchors), hnorm(target_hidden))).
    {
        const anchor = try g.slice(x, &.{ 0, 0 }, &.{ 1, hidden });
        const e_n = try g.rmsNorm(anchor, d.enorm, d.rms_eps);
        const h_n = try g.rmsNorm(target_hidden, d.hnorm, d.rms_eps);
        const cat = try g.concat(&.{ e_n, h_n }, -1);
        const fused = try d.eh_proj.forward(&g, cat);
        const gated = try g.mul(fused, d.fusion_gate);
        const slot0 = try g.add(anchor, gated);
        const rest = try g.slice(x, &.{ 1, 0 }, &.{ @intCast(B), hidden });
        x = try g.concat(&.{ slot0, rest }, 0);
    }

    for (d.layers, 0..) |*l, n| {
        const target = if (n < TargetLayers.len) dfly_mod.TargetLayers[n] else 1;
        const normed = try g.rmsNorm(x, l.input_norm, d.rms_eps);
        const attn = try dflyAttn(&g, m, d, l, normed, target, positions);
        x = try g.add(x, attn);
        const normed2 = try g.rmsNorm(x, l.post_attn_norm, d.rms_eps);
        const gg = try l.gate.forward(&g, normed2);
        const uu = try l.up.forward(&g, normed2);
        const act = try g.mul(try g.silu(gg), uu);
        const dn = try l.down.forward(&g, act);
        x = try g.add(x, dn);
    }

    var z = try g.rmsNorm(x, d.final_norm, d.rms_eps); // [B,hidden]
    if (d.c.dfly_layers == 0) return error.NoDrafter;

    var tokens: [abi.max_round_tokens]u32 = @splat(0);
    if (B > abi.max_round_tokens) return error.TooManyRows;
    try ops.eval(ids);
    const idp = mlx.mlx_array_data_int32(ids) orelse return error.NoData;
    var prev: u32 = @intCast(idp[0]);
    for (1..B) |i| {
        const row = try g.slice(z, &.{ @intCast(i), 0 }, &.{ @intCast(i + 1), hidden });
        const h_n = try g.rmsNorm(row, d.hc_hidden_norm, d.rms_eps);
        const prev_arr = try g.fromSlice(i32, &[_]i32{@intCast(prev)}, &.{ 1, 1 });
        const prev_e_raw = try m.embedRows(&g, prev_arr);
        const prev_e = try g.reshape(prev_e_raw, &.{ 1, hidden });
        const e_n = try g.rmsNorm(prev_e, d.hc_embed_norm, d.rms_eps);
        const cat = try g.concat(&.{ h_n, e_n }, -1);
        const gg = try d.hc_gate.forward(&g, cat);
        const uu = try d.hc_up.forward(&g, cat);
        const act = try g.mul(try g.silu(gg), uu);
        const dn = try d.hc_down.forward(&g, act);
        const corrected = try g.add(row, dn);
        const logits = try m.lm_head_row(&g, corrected); // [1,vocab]
        const tok = try argmaxMasked(&g, logits, null);
        tokens[i - 1] = tok;
        prev = tok;
        // Build the corrected row back into z (only this row is used next).
        const before = try g.slice(z, &.{ 0, 0 }, &.{ @intCast(i), hidden });
        z = try g.concat(&.{ before, corrected, try g.slice(z, &.{ @intCast(i + 1), 0 }, &.{ @intCast(B), hidden }) }, 0);
    }
    return tokens[0 .. B - 1];
}

fn dflyAttn(g: *Graph, m: *Model, d: *dfly_mod.Drafter, l: *const dfly_mod.Layer, x: Array, target: u32, positions: []const u32) !Array {
    const B: c_int = @intCast(ops.shape(x)[0]);
    const heads: c_int = @intCast(l.heads);
    const nope: c_int = @intCast(l.nope);
    const rope_dim: c_int = @intCast(l.rope_dim);
    const vd: c_int = @intCast(l.v_dim);
    const rank: c_int = @intCast(l.rank);
    const qk = nope + rope_dim;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(qk)));
    var t = g.scope();
    defer t.deinit();

    const q_resid = try t.rmsNorm(try l.q_a.forward(&t, x), l.q_a_norm, d.rms_eps);
    const q_flat = try l.q_b.forward(&t, q_resid);
    const q4 = try t.transpose(try t.reshape(q_flat, &.{ 1, B, heads, qk }), &.{ 0, 2, 1, 3 });
    const q_nope = try t.slice(q4, &.{ 0, 0, 0, 0 }, &.{ 1, heads, B, nope });
    const q_rot = try t.slice(q4, &.{ 0, 0, 0, nope }, &.{ 1, heads, B, qk });
    const kv_a = try l.kv_a.forward(&t, x);
    const lat_raw = try t.slice(kv_a, &.{ 0, 0 }, &.{ B, rank });
    const k_rot = try t.slice(kv_a, &.{ 0, rank }, &.{ B, rank + rope_dim });
    const latent = try t.rmsNorm(lat_raw, l.kv_a_norm, d.rms_eps);
    const off: c_int = @intCast(positions[0]);
    const q_rot_r = try t.call(mlx.mlx_fast_rope, .{ q_rot, rope_dim, true, mlx.mlx_optional_float.some(l.theta), 1.0, off, Array{} });
    const k_rot3 = try t.reshape(k_rot, &.{ 1, B, rope_dim });
    const k_rot_r3 = try t.call(mlx.mlx_fast_rope, .{ k_rot3, rope_dim, true, mlx.mlx_optional_float.some(l.theta), 1.0, off, Array{} });
    const k_rot_r = try t.reshape(k_rot_r3, &.{ B, rope_dim });
    const q_abs = try t.matmul(q_nope, l.wk); // [1,heads,B,rank]
    const q_cat = try t.concat(&.{ q_abs, q_rot_r }, -1);

    // Target KV history for this draft layer's target, then the block's own rows.
    var view: abi.KvView = .{};
    const vop = m.slotCacheOps() orelse return error.NoCacheOps;
    const v = vop.view(m.slotCacheHandle() orelse return error.NoCache, target, .{ .ctx = d.s.ctx }, &view);
    if (!v.isOk()) return error.KvViewFailed;
    defer vop.release(&view);
    if (view.k.ctx == null or view.v.ctx == null) return error.EmptyKvView;
    const hk = try t.astype(.{ .ctx = view.k.ctx }, ops.dtype(x));
    const hv = try t.astype(.{ .ctx = view.v.ctx }, ops.dtype(x));
    const hlen = ops.shape(hk)[2];
    const keys_h = try t.slice(hk, &.{ 0, 0, 0, 0 }, &.{ 1, 1, hlen, rank + rope_dim });
    const own_k = try t.reshape(try t.concat(&.{ latent, k_rot_r }, -1), &.{ 1, 1, B, rank + rope_dim });
    const own_v = try t.reshape(latent, &.{ 1, 1, B, rank });
    const all_k = try t.concat(&.{ keys_h, own_k }, 2);
    const all_v = try t.concat(&.{ hv, own_v }, 2);
    const kv_len: c_int = hlen + B;
    const mask = try sliding.causalMask(g, B, kv_len);
    const sinks = try t.astype(l.sink, ops.dtype(x));
    var o_lat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_lat, q_cat, all_k, all_v, scale, "array", mask, sinks, false, t.s));
    const o_arr = try t.adopt(o_lat);
    const wv_t = try t.transpose(l.wv, &.{ 0, 2, 1 });
    const values = try t.matmul(o_arr, wv_t);
    const flat = try t.reshape(try t.transpose(values, &.{ 0, 2, 1, 3 }), &.{ 1, B, heads * vd });
    const gate = try t.sigmoid(try l.linear_gate.forward(&t, x));
    const out = try t.rmsNorm(try l.o_proj.forward(&t, try t.mul(flat, gate)), l.post_proj_norm, d.rms_eps);
    return g.adopt(t.escape(out));
}

fn argmaxMasked(g: *Graph, logits: Array, mask: ?Array) !u32 {
    var l = logits;
    if (mask) |m| {
        const neg = try g.scalarLike(logits, -std.math.inf(f32));
        l = try g.where(m, neg, logits);
    }
    const idx = try g.argmax(l, -1, false);
    try ops.eval(idx);
    const p = mlx.mlx_array_data_int32(idx) orelse return error.NoData;
    return @intCast(p[0]);
}

pub const hooks: abi.NativeDrafter = .{ .step = &stepHook, .verify = &verifyHook, .finish = &finishHook };


fn dflyModelOf(m: *anyopaque) *Model {
    const wm: *weights.Model = @ptrCast(@alignCast(m));
    return wm.trunk;
}

fn dflySlotState(m: *Model, f: *const abi.Forward) !*SlotState {
    if (f.slots.len == 0) return error.NoSlot;
    const sid: usize = f.slots.slice()[0];
    if (sid >= m.slot_state.len) return error.NoSlot;
    const st = &m.slot_state[sid];
    if (!st.active or st.hidden.ctx == null or st.cache == null or st.cache_ops == null) return error.NoDrafterState;
    return st;
}

fn stepHook(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const mdl = dflyModelOf(m);
    const d = mdl.dfly orelse return Verdict.refuse(.unsupported, "hy_v4_flash: no DFly2 drafter loaded");
    const st = dflySlotState(mdl, f) catch return Verdict.refuse(.failed, "hy_v4_flash: no target state for the draft round");
    const t1: u32 = if (f.tokens.len > 0) f.tokens.slice()[0] else return Verdict.refuse(.failed, "hy_v4_flash: a round needs t1");
    const anchor: u32 = if (f.positions.len > 0) f.positions.slice()[0] else return Verdict.refuse(.failed, "hy_v4_flash: a round needs a position");
    const cap: usize = @min(@as(usize, f.accept_cap), 5);
    if (cap == 0) return Verdict.refuse(.failed, "hy_v4_flash: no draft budget");
    const B: usize = cap + 1;
    const ids = gpa.alloc(i32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(ids);
    ids[0] = @intCast(t1);
    for (1..B) |i| ids[i] = @intCast(d.mask_id);
    const ids_arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(B) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const positions = gpa.alloc(u32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(positions);
    for (positions, 0..) |*p, i| p.* = anchor + @as(u32, @intCast(i));
    const drafts = dflyDraft(mdl, d, .{ .ctx = ids_arr.ctx }, positions, .{ .ctx = st.hidden.ctx }) catch |e| {
        log.err("hy_v4 dfly: draft failed: {t}\n", .{e});
        return Verdict.named(.failed, @errorName(e));
    };
    defer gpa.free(drafts);

    var verify_ids = gpa.alloc(u32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(verify_ids);
    verify_ids[0] = t1;
    for (drafts, 0..) |tok, i| verify_ids[i + 1] = tok;
    const slot_ids = [_]u32{@intCast(f.slots.slice()[0])};
    var vf: abi.Forward = .{
        .mode = .verify,
        .rows = @intCast(B),
        .tokens = .of(verify_ids),
        .slots = .of(&slot_ids),
        .positions = .of(positions),
        .caches = .of(&[_]abi.Handle{st.cache.?}),
        .cache_ops = st.cache_ops,
        .logits = .all_rows,
        .want_hidden = true,
    };
    const vout = mdl.forward(&vf) catch |e| {
        log.err("hy_v4 dfly: verify failed: {t}\n", .{e});
        return Verdict.named(.failed, @errorName(e));
    };
    defer _ = mlx.mlx_array_free(.{ .ctx = vout.logits.ctx });
    defer _ = mlx.mlx_array_free(.{ .ctx = vout.hidden.ctx });

    // Accept the drafts the target's argmax agrees with, in order.
    const logits0: mlx.mlx_array = .{ .ctx = vout.logits.ctx };
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    if (mlx.mlx_astype(&logits, logits0, .float32, d.s) != 0) return Verdict.refuse(.failed, "hy_v4_flash: logits cast failed");
    if (mlx.mlx_array_eval(logits) != 0) return Verdict.refuse(.failed, "hy_v4_flash: logits eval failed");
    const lp = mlx.mlx_array_data_float32(logits) orelse return Verdict.refuse(.failed, "hy_v4_flash: no logits");
    var accepted: u32 = 0;
    for (0..B - 1) |i| {
        const row_argmax = argmaxRow32(lp, i, @intCast(d.c.vocab_size), f.suppress_mask);
        if (row_argmax != verify_ids[i + 1]) break;
        accepted += 1;
    }
    const next_token = argmaxRow32(lp, accepted, @intCast(d.c.vocab_size), f.suppress_mask);

    const keep = anchor + 1 + accepted;
    const trunc = st.cache_ops.?.truncateOp() orelse return Verdict.refuse(.unsupported, "hy_v4_flash: the host cache has no truncate");
    if (!trunc(st.cache.?, keep, .{ .ctx = d.s.ctx }).isOk()) return Verdict.refuse(.failed, "hy_v4_flash: cache rollback failed");
    dflySettle(mdl, st, vout.hidden, accepted, anchor, verify_ids, B);

    out.round = .{ .n = accepted + 1, .accepted = accepted, .next_token = next_token, .rows = @intCast(B), .position = keep };
    for (verify_ids[0..accepted + 1], 0..) |tok, i| out.round.tokens[i] = tok;
    return .ok;
}

fn verifyHook(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const mdl = dflyModelOf(m);
    const d = mdl.dfly orelse return Verdict.refuse(.unsupported, "hy_v4_flash: no DFly2 drafter loaded");
    const st = dflySlotState(mdl, f) catch return Verdict.refuse(.failed, "hy_v4_flash: no target state for the draft round");
    const t1: u32 = if (f.tokens.len > 0) f.tokens.slice()[0] else return Verdict.refuse(.failed, "hy_v4_flash: a round needs t1");
    const anchor: u32 = if (f.positions.len > 0) f.positions.slice()[0] else return Verdict.refuse(.failed, "hy_v4_flash: a round needs a position");
    const cap: usize = @min(@as(usize, f.accept_cap), 5);
    if (cap == 0) return Verdict.refuse(.failed, "hy_v4_flash: no draft budget");
    const B: usize = cap + 1;
    const ids = gpa.alloc(i32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(ids);
    ids[0] = @intCast(t1);
    for (1..B) |i| ids[i] = @intCast(d.mask_id);
    const ids_arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(B) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const positions = gpa.alloc(u32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(positions);
    for (positions, 0..) |*p, i| p.* = anchor + @as(u32, @intCast(i));
    const drafts = dflyDraft(mdl, d, .{ .ctx = ids_arr.ctx }, positions, .{ .ctx = st.hidden.ctx }) catch |e| {
        log.err("hy_v4 dfly: draft failed: {t}\n", .{e});
        return Verdict.named(.failed, @errorName(e));
    };
    defer gpa.free(drafts);
    var verify_ids = gpa.alloc(u32, B) catch return Verdict.refuse(.failed, "hy_v4_flash: out of memory");
    defer gpa.free(verify_ids);
    verify_ids[0] = t1;
    for (drafts, 0..) |tok, i| verify_ids[i + 1] = tok;
    const slot_ids = [_]u32{@intCast(f.slots.slice()[0])};
    var vf: abi.Forward = .{
        .mode = .verify,
        .rows = @intCast(B),
        .tokens = .of(verify_ids),
        .slots = .of(&slot_ids),
        .positions = .of(positions),
        .caches = .of(&[_]abi.Handle{st.cache.?}),
        .cache_ops = st.cache_ops,
        .logits = .all_rows,
        .want_hidden = true,
    };
    const vout = mdl.forward(&vf) catch |e| {
        log.err("hy_v4 dfly: verify failed: {t}\n", .{e});
        return Verdict.named(.failed, @errorName(e));
    };
    if (st.verify_hidden.ctx != null) _ = mlx.mlx_array_free(st.verify_hidden);
    st.verify_hidden = .{ .ctx = vout.hidden.ctx };
    st.verify_n = @intCast(B);
    st.verify_open = true;
    st.anchor = anchor;
    for (verify_ids, 0..) |tok, i| st.verify_tokens[i] = tok;
    out.logits = vout.logits;
    out.round = .{ .n = @intCast(B), .accepted = 0, .next_token = verify_ids[B - 1], .rows = @intCast(B), .position = anchor + @as(u64, B) };
    for (verify_ids, 0..) |tok, i| out.round.tokens[i] = tok;
    return .ok;
}

fn finishHook(m: *anyopaque, accepted: u32, out: *abi.ForwardOut) callconv(.c) Verdict {
    const mdl = dflyModelOf(m);
    const st = if (mdl.active_slot < mdl.slot_state.len) &mdl.slot_state[mdl.active_slot] else return Verdict.refuse(.failed, "hy_v4_flash: no slot");
    if (!st.verify_open or st.cache == null or st.cache_ops == null) return Verdict.refuse(.failed, "hy_v4_flash: no open verify");
    const keep = st.anchor + 1 + accepted;
    const trunc = st.cache_ops.?.truncateOp() orelse return Verdict.refuse(.unsupported, "hy_v4_flash: the host cache has no truncate");
    if (!trunc(st.cache.?, keep, .{ .ctx = mlx.gpuStream().ctx }).isOk()) return Verdict.refuse(.failed, "hy_v4_flash: cache rollback failed");
    const hidden: mlx.mlx_array = .{ .ctx = st.verify_hidden.ctx };
    const rows = @as(c_int, @intCast(st.verify_n));
    const hdim: c_int = @intCast(mdl.c.hidden_size);
    if (st.hidden.ctx != null) _ = mlx.mlx_array_free(st.hidden);
    var sl = mlx.mlx_array_new();
    if (mlx.mlx_slice(&sl, hidden, &[_]c_int{ 0, @intCast(@min(accepted, st.verify_n - 1)), 0 }, 3, &[_]c_int{ 1, @intCast(@min(accepted, st.verify_n - 1) + 1), hdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()) != 0) return Verdict.refuse(.failed, "hy_v4_flash: hidden slice failed");
    st.hidden = .{ .ctx = sl.ctx };
    _ = rows;
    st.verify_open = false;
    out.round = .{ .position = keep };
    return .ok;
}

fn argmaxRow32(lp: [*]const f32, row: usize, vocab: usize, suppress: abi.Array) u32 {
    const base = row * vocab;
    var best: u32 = 0;
    var bestv: f32 = -std.math.inf(f32);
    if (suppress.ctx != null) {
        const sm = mlx.mlx_array_data_uint8(.{ .ctx = suppress.ctx }) orelse {
            for (0..vocab) |i| if (lp[base + i] > bestv) {
                bestv = lp[base + i];
                best = @intCast(i);
            };
            return best;
        };
        for (0..vocab) |i| {
            if (sm[i] != 0) continue;
            if (lp[base + i] > bestv) {
                bestv = lp[base + i];
                best = @intCast(i);
            }
        }
        return best;
    }
    for (0..vocab) |i| if (lp[base + i] > bestv) {
        bestv = lp[base + i];
        best = @intCast(i);
    };
    return best;
}

fn dflySettle(mdl: *Model, st: *SlotState, hidden_out: abi.Array, accepted: u32, anchor: u32, tokens: []const u32, n: usize) void {
    const hidden: mlx.mlx_array = .{ .ctx = hidden_out.ctx };
    const hdim: c_int = @intCast(mdl.c.hidden_size);
    if (st.hidden.ctx != null) _ = mlx.mlx_array_free(st.hidden);
    var sl = mlx.mlx_array_new();
    if (mlx.mlx_slice(&sl, hidden, &[_]c_int{ 0, @intCast(@min(accepted, n - 1)), 0 }, 3, &[_]c_int{ 1, @intCast(@min(accepted, n - 1) + 1), hdim }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()) != 0) return;
    _ = mlx.mlx_array_eval(sl);
    st.hidden = .{ .ctx = sl.ctx };
    _ = anchor;
    _ = tokens;
    st.verify_open = false;
}
