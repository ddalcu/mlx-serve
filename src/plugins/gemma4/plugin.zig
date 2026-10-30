//! Gemma 4 text decoders (dense): `decoder.zig`'s skeleton over the Gemma attention and GeGLU blocks, per-layer
//! embeddings and layer scalars. The load reads the engine's tensors through the pack's quant format (Gemma 3's
//! loader); the drafter hook hands the Gemma 4 assistant the trunk's embedding table and head.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const Array = mlx.mlx_array;
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
const native = @import("../../sdk/quant/native.zig");
const load_mod = @import("../gemma3/load.zig");
const compiled_mod = @import("../gemma3/compiled.zig");
const moe = @import("moe.zig");
const gemma = @import("../gemma3/config.zig");
const bases_compiled = @import("../../sdk/bases/blocks/compiled.zig");
const ops = @import("../../mlx/ops.zig");
const quantParamsFromGeometry = @import("../../sdk/quant/params.zig").quantParamsFromGeometry;
pub const config = @import("config.zig");
pub const decoder = @import("decoder.zig");

pub const plugin = config.plugin;

comptime {
    if (decoder.batches_decode != config.batches_decode) @compileError("gemma4: config.batches_decode disagrees with the decoder");
}

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &config.layout },
    .spec = &.{ .trunk_weight = &load_mod.TrunkWeight(Loaded).get },
    .weights = &.{},
};

const gpa = load_mod.gpa;
const Verdict = abi.Verdict;
const Linear = bases.blocks.linear.Linear;

/// Every term is the engine's default arithmetic over the cache layout.
pub fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(decoder),
    layers: []decoder.Layer = &.{},
    owned: load_mod.Owned = .{},
    angles: qk_rope.AngleCache = .{},
    compiled: compiled_mod.Compiled = undefined,
    /// The experts' routing chain compiled (packs with experts, on a GPU).
    routing: bases_compiled.Closure = .{ .ctx = null },

    fn release(m: *Loaded) void {
        bases_compiled.free(&m.routing);
        m.angles.deinit();
        m.compiled.deinit();
        m.owned.deinit();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

pub fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try load_mod.Reader.open(ctx, c);
    defer r.close();
    const s = r.s;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids } };
    m.compiled.init(s, c.final_logit_softcapping);
    errdefer m.release();
    if (c.num_experts > 0) m.routing = moe.compileRouting(@intCast(c.num_experts_per_tok), s);
    m.layers = try gpa.alloc(decoder.Layer, c.num_hidden_layers);

    const o = &m.owned;
    const sh = try load_mod.AttnShared.build(o, c, s);
    const ple_dim = c.hidden_size_per_layer_input;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        const li: u32 = @intCast(i);
        // Gemma 4's q/k norms carry the scale: attention runs at 1.0.
        const attn = try load_mod.attentionAt(&r, o, c, &sh, li, 1.0, &m.angles);
        const scalar = try r.share(try r.name(&nb, "layers.{d}.layer_scalar", .{li}));
        try o.own(scalar);
        L.* = .{
            .input_norm = try r.norm(o, try r.name(&nb, "layers.{d}.input_layernorm.weight", .{li}), false),
            .attn = attn,
            .post_attn_norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{li}), false),
            .pre_ff_norm = try r.norm(o, try r.name(&nb, "layers.{d}.pre_feedforward_layernorm.weight", .{li}), false),
            // A 26B-A4B layer's shared MLP joins gate|up whenever it is quantized, as the layered load does.
            .mlp = try load_mod.mlpAt(&r, o, c, li, c.num_experts > 0 or attn.qkv.built(), &m.compiled),
            .post_ff_norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_feedforward_layernorm.weight", .{li}), false),
            .layer_scalar = scalar,
            .ple = if (ple_dim > 0 and r.has(try r.name(&nb, "layers.{d}.per_layer_input_gate.weight", .{li}))) .{
                .gate = try ownedProj(&r, o, try r.name(&nb, "layers.{d}.per_layer_input_gate", .{li}), c.hidden_size),
                .proj = try ownedProj(&r, o, try r.name(&nb, "layers.{d}.per_layer_projection", .{li}), ple_dim),
                .norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_per_layer_input_norm.weight", .{li}), false),
            } else null,
            .moe = if (c.num_experts > 0) try moeAt(&r, o, c, li, &m.compiled, m.routing) else null,
        };
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try o.linear(embed);
    const emb_scale = load_mod.bf16Scalar(@sqrt(@as(f32, @floatFromInt(c.hidden_size))), s);
    try o.own(emb_scale);
    var w: decoder.Weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = .bfloat16, .scale = emb_scale },
        .layers = m.layers,
        .final_norm = try r.norm(o, try r.name(&nb, "norm.weight", .{}), false),
        .head = .{
            .head = try load_mod.headOf(&r, o, embed, c.hidden_size, c.tie_word_embeddings),
            .cap = m.compiled.cap,
            .compiled = m.compiled.softcap,
        },
        .compiled = &m.compiled,
    };
    if (ple_dim > 0) {
        w.ple_dim = @intCast(ple_dim);
        w.ple_table = try r.tableAt(try r.name(&nb, "embed_tokens_per_layer", .{}), @intCast(c.num_hidden_layers * ple_dim));
        try o.linear(w.ple_table);
        w.ple_proj = try r.tableAt(try r.name(&nb, "per_layer_model_projection", .{}), c.hidden_size);
        try o.linear(w.ple_proj);
        w.ple_proj_norm = try r.norm(o, try r.name(&nb, "per_layer_projection_norm.weight", .{}), false);
    }
    m.base.weights = w;
    if (gemma.ternaryCandidate(c)) try load_mod.armTernary(.{ &m.layers, &m.base.weights.head.head, &m.base.weights.ple_table, &m.base.weights.ple_proj }, null, load_mod.gatherOnly(&m.base.weights.head.head, embed), s);
    return m;
}

/// Layer `li`'s routed half: the router (its scale folded with `hidden^-0.5`, as the engine folds it), the expert
/// banks (`experts.switch_glu.*`, or a packed `experts.gate_up_proj` split gate rows first).
pub fn moeAt(r: *load_mod.Reader, o: *load_mod.Owned, c: *const config.Config, li: u32, compiled: *const compiled_mod.Compiled, routing: bases_compiled.Closure) !decoder.MoeTail {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const s = r.s;
    const rs = try r.need(try r.name(&nb, "layers.{d}.router.scale", .{li}));
    defer _ = mlx.mlx_array_free(rs);
    const root = load_mod.bf16Scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hidden))), s);
    defer _ = mlx.mlx_array_free(root);
    var folded = mlx.mlx_array_new();
    mlx.check(mlx.mlx_multiply(&folded, rs, root, s)) catch |e| {
        _ = mlx.mlx_array_free(folded);
        return e;
    };
    try o.own(folded);
    const pes = try r.share(try r.name(&nb, "layers.{d}.router.per_expert_scale", .{li}));
    if (pes.ctx != null) try o.own(pes);
    var gate: moe.Bank = undefined;
    var up: moe.Bank = undefined;
    if (r.has(try r.name(&nb, "layers.{d}.experts.gate_up_proj.weight", .{li}))) {
        const halves = try packedBanks(r, o, try r.name(&nb, "layers.{d}.experts.gate_up_proj", .{li}), hidden);
        gate = halves[0];
        up = halves[1];
    } else {
        gate = try bankAt(r, o, try r.name(&nb, "layers.{d}.experts.switch_glu.gate_proj", .{li}), hidden);
        up = try bankAt(r, o, try r.name(&nb, "layers.{d}.experts.switch_glu.up_proj", .{li}), hidden);
    }
    const down_base = if (r.has(try r.name(&nb, "layers.{d}.experts.down_proj.weight", .{li})))
        try r.name(&nb, "layers.{d}.experts.down_proj", .{li})
    else
        try r.name(&nb, "layers.{d}.experts.switch_glu.down_proj", .{li});
    const down = try bankAt(r, o, down_base, c.moe_intermediate_size);
    return .{
        .pre_ff_norm_2 = try r.norm(o, try r.name(&nb, "layers.{d}.pre_feedforward_layernorm_2.weight", .{li}), false),
        .post_ff_norm_1 = try r.norm(o, try r.name(&nb, "layers.{d}.post_feedforward_layernorm_1.weight", .{li}), false),
        .post_ff_norm_2 = try r.norm(o, try r.name(&nb, "layers.{d}.post_feedforward_layernorm_2.weight", .{li}), false),
        .experts = .{
            .router = try ownedProj(r, o, try r.name(&nb, "layers.{d}.router.proj", .{li}), hidden),
            .router_scale = folded,
            .per_expert_scale = pes,
            .gate = gate,
            .up = up,
            .down = down,
            .top_k = @intCast(c.num_experts_per_tok),
            .experts = @intCast(c.num_experts),
            .route_fn = routing,
            .compiled = compiled,
        },
    };
}

fn bankAt(r: *load_mod.Reader, o: *load_mod.Owned, base: []const u8, in_dim: u32) !moe.Bank {
    const l = try ownedProj(r, o, base, in_dim);
    return .{ .w = l.w, .s = l.s, .b = l.b, .qp = .{ .bits = l.bits, .group_size = l.group_size, .mode = l.mode } };
}

/// A packed `[E, 2M, in]` bank (and its scales and biases) as two contiguous halves, gate rows first, as the engine's
/// `splitPackedGateUp` makes them; a dense pack's halves are transposed after the split.
fn packedBanks(r: *load_mod.Reader, o: *load_mod.Owned, base: []const u8, in_dim: u32) ![2]moe.Bank {
    var nb: [256]u8 = undefined;
    var parts: [3][2]Array = .{ .{ .{}, .{} }, .{ .{}, .{} }, .{ .{}, .{} } };
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, k| {
        const raw = try r.share(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ base, leaf }));
        if (raw.ctx == null) continue;
        defer _ = mlx.mlx_array_free(raw);
        const sh = mlx.getShape(raw);
        if (sh.len != 3 or @rem(sh[1], 2) != 0) return error.BadPackedGateUpShape;
        const m_rows = @divExact(sh[1], 2);
        for (0..2) |half| {
            var view = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(view);
            const lo: c_int = @intCast(half * @as(usize, @intCast(m_rows)));
            try mlx.check(mlx.mlx_slice(&view, raw, &[_]c_int{ 0, lo, 0 }, 3, &[_]c_int{ sh[0], lo + m_rows, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, r.s));
            var part = mlx.mlx_array_new();
            mlx.check(mlx.mlx_contiguous(&part, view, false, r.s)) catch |e| {
                _ = mlx.mlx_array_free(part);
                return e;
            };
            try o.own(part);
            parts[k][half] = part;
        }
    }
    if (parts[0][0].ctx == null) return error.MissingWeight;
    var out: [2]moe.Bank = undefined;
    for (&out, 0..) |*bk, half| {
        const w = parts[0][half];
        const sc = parts[1][half];
        if (sc.ctx == null) {
            var t = mlx.mlx_array_new();
            mlx.check(mlx.mlx_transpose_axes(&t, w, &[_]c_int{ 0, 2, 1 }, 3, r.s)) catch |e| {
                _ = mlx.mlx_array_free(t);
                return e;
            };
            try o.own(t);
            bk.* = .{ .w = t, .qp = .{ .bits = 0, .group_size = 0 } };
            continue;
        }
        const qp = quantParamsFromGeometry(w, sc, parts[2][half].ctx != null, in_dim) orelse return error.PackedExpertsUnknownQuant;
        bk.* = .{ .w = w, .s = sc, .b = parts[2][half], .qp = qp };
    }
    return out;
}

fn ownedProj(r: *load_mod.Reader, o: *load_mod.Owned, base: []const u8, in_dim: u32) !Linear {
    const l = try r.proj(base, in_dim);
    try o.linear(l);
    return l;
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    // Packs with experts decode serial: their parse declares no batched rows.
    if (f.mode == .batched_rows and m.layers.len > 0 and m.layers[0].moe != null)
        return Verdict.refuse(.unsupported, "forward mode not declared");
    return run_mod.tableForward(decoder, &m.base, f, out);
}


/// The tiny pack the conformance suite runs: four layers (sliding, global of a wider head, sliding, a KV-shared
/// global reading layer 1), per-layer embeddings, layer scalars, value norms, a softcap; seeded random f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const hd = 32;
    const ghd = 64;
    const pd = 8;
    const n_layers = 4;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 7;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = try al.dupe(u64, shape), .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    const p = "language_model.model.";
    try put(a, &tensors, &seed, p ++ "embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, p ++ "norm.weight", &.{h}, 0.1);
    try put(a, &tensors, &seed, p ++ "embed_tokens_per_layer.weight", &.{ v, n_layers * pd }, 0.5);
    try put(a, &tensors, &seed, p ++ "per_layer_model_projection.weight", &.{ n_layers * pd, h }, 0.1);
    try put(a, &tensors, &seed, p ++ "per_layer_projection_norm.weight", &.{pd}, 0.1);
    for (0..n_layers) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "language_model.model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        const global = i % 2 == 1;
        const d: u64 = if (global) ghd else hd;
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "post_per_layer_input_norm.weight" }) |nm|
            try put(a, &tensors, &seed, try L.n(a, i, nm), &.{h}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "layer_scalar"), &.{1}, 0.5);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_norm.weight"), &.{d}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_proj.weight"), &.{ 2 * d, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.o_proj.weight"), &.{ h, 2 * d }, 0.1);
        if (i < 3) {
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_norm.weight"), &.{d}, 0.1);
            try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_proj.weight"), &.{ d, h }, 0.1);
            if (!global) try put(a, &tensors, &seed, try L.n(a, i, "self_attn.v_proj.weight"), &.{ d, h }, 0.1);
        }
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.down_proj.weight"), &.{ h, inter }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "per_layer_input_gate.weight"), &.{ pd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "per_layer_projection.weight"), &.{ h, pd }, 0.1);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"gemma4_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"global_head_dim":64,
        \\ "num_global_key_value_heads":1,"num_kv_shared_layers":1,"attention_k_eq_v":true,"vocab_size":64,
        \\ "hidden_size_per_layer_input":8,"rms_norm_eps":1e-6,"sliding_window":2,"tie_word_embeddings":true,
        \\ "layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
        \\ "rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
        \\ "sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"max_position_embeddings":4096,
        \\ "eos_token_id":[1,106]}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
    _ = @import("contract.zig");
}
