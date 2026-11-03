//! LFM2 (`lfm2`, `lfm2_moe`): the hybrid base over the engine's loaded tensors. Each layer is a gated short
//! convolution or q/k-normed attention, then a SwiGLU (or, past `num_dense_layers` on lfm2_moe, a sigmoid-routed
//! MoE) under `ffn_norm`, closing on `residualTail` seams; `embedding_norm` is the final norm.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const reader = @import("../../sdk/bases/reader.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// One of two MLP blocks per layer (the dense SwiGLU or the MoE): `block.Either` for the MLP slot.
pub fn EitherMlp(comptime A: type, comptime B: type) type {
    block.check(A, &.{.mlp});
    block.check(B, &.{.mlp});
    return struct {
        pub const kind: block.Kind = .mlp;
        pub const supports_batched_rows = A.supports_batched_rows and B.supports_batched_rows;
        pub const Weights = union(enum) { first: A.Weights, second: B.Weights };

        pub fn forward(w: *const Weights, c: *block.Ctx, x: Array) !Array {
            return switch (w.*) {
                .first => |*a| A.forward(a, c, x),
                .second => |*m| B.forward(m, c, x),
            };
        }
    };
}

pub const Mixer = bases.Either(b.gated_conv.GatedConv, b.attention.HybridAttention);
pub const Mlp = EitherMlp(b.mlp.GatedMlp, moe.Moe);

pub const Model = bases.Hybrid(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .op = Mixer,
    .mlp = Mlp,
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
    // The engine's LFM2 publishes the residual, not the final-normed hidden.
    .hooks = struct {
        pub const residual_hidden = true;
    },
});

comptime {
    if (Model.batches_decode != config.batches_decode) @compileError("lfm2: config.batches_decode disagrees with the base");
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
    .cache = &.{ .layout = config.layout },
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded) },
    .weights = &.{},
};

/// Every term is the engine's default arithmetic over the shell and the layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A loaded model: the base's weights and every array handle it holds.
pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},

    fn release(m: *Loaded) void {
        m.owned.deinit();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    const prefix = try reader.probePrefix(ctx, c.prefix, &.{ "language_model.model", "model.language_model", "model" }, "embed_tokens.weight");
    var r = try reader.Reader.open(ctx, c.block_json, prefix);
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(m, &r, c, @intCast(i));

    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const embed = try r.table(try r.name(&nb, "embed_tokens", .{}), hidden);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "embedding_norm.weight", .{}));
    try m.owned.own(final_norm);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = try headOf(&r, m, embed, hidden, c.tie_word_embeddings),
    };
    // The engine's ternary arm (qmv2's kernels) for a 2-bit group-128 pack whose every bias is -scale.
    if (c.quant.mode == .affine and c.quant.bits == 2 and c.quant.group_size == 128)
        try reader.armTernary(.{ &m.layers, &m.base.weights.head }, null, reader.gatherOnly(&m.base.weights.head, embed), r.s);
    return m;
}

fn loadLayer(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !Model.Layer {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const s = r.s;
    const input_norm = try r.need(try r.name(&nb, "layers.{d}.operator_norm.weight", .{i}));
    try m.owned.own(input_norm);
    const post_norm = try r.need(try r.name(&nb, "layers.{d}.ffn_norm.weight", .{i}));
    try m.owned.own(post_norm);
    const op: Mixer.Weights = switch (c.layers[i]) {
        .conv => blk: {
            const in_proj = try r.proj(try r.name(&nb, "layers.{d}.conv.in_proj", .{i}), hidden);
            try m.owned.linear(in_proj);
            const conv = try reader.depthwiseConv(try r.need(try r.name(&nb, "layers.{d}.conv.conv.weight", .{i})), s);
            try m.owned.own(conv);
            const out_proj = try r.proj(try r.name(&nb, "layers.{d}.conv.out_proj", .{i}), hidden);
            try m.owned.linear(out_proj);
            break :blk .{ .first = .{ .in_proj = in_proj, .conv = conv, .out_proj = out_proj, .hidden = @intCast(hidden), .kernel = @intCast(c.conv_kernel) } };
        },
        .attention => blk: {
            var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
            var k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
            var v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
            const o = try r.proj(try r.name(&nb, "layers.{d}.self_attn.out_proj", .{i}), c.num_attention_heads * c.head_dim);
            const qkv = if (q.s.ctx != null) try reader.join(&m.owned, &.{ &q, &k, &v }, s) else b.linear.Joined{};
            for ([_]Linear{ q, k, v, o }) |l| try m.owned.linear(l);
            const q_norm = try r.share(try r.name(&nb, "layers.{d}.self_attn.q_layernorm.weight", .{i}));
            try m.owned.own(q_norm);
            const k_norm = try r.share(try r.name(&nb, "layers.{d}.self_attn.k_layernorm.weight", .{i}));
            try m.owned.own(k_norm);
            break :blk .{ .second = .{
                .q = q,
                .k = k,
                .v = v,
                .o = o,
                .q_norm = q_norm,
                .k_norm = k_norm,
                .qkv = qkv,
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
                    .rope_dims = @intCast(c.head_dim),
                    .rope_base = c.rope_theta,
                    .rope_scale = 1.0,
                },
                .keep = c.max_position_embeddings,
            } };
        },
    };
    return .{ .input_norm = input_norm, .op = op, .post_norm = post_norm, .mlp = try mlpOf(m, r, c, i) };
}

fn mlpOf(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !Mlp.Weights {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    if (c.moeAt(i)) |mc| {
        const router = try r.proj(try r.name(&nb, "layers.{d}.feed_forward.gate", .{i}), hidden);
        try m.owned.linear(router);
        var banks: [3]Linear = undefined;
        for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, mc.intermediate }) |*bk, leaf, in_dim| {
            bk.* = try r.proj(try r.name(&nb, "layers.{d}.feed_forward.switch_mlp.{s}", .{ i, leaf }), in_dim);
            try m.owned.linear(bk.*);
        }
        // `use_expert_bias: false` ships none; the reference zero-inits it (plain sigmoid top-k).
        var bias = try r.share(try r.name(&nb, "layers.{d}.feed_forward.expert_bias", .{i}));
        if (bias.ctx == null) bias = try reader.zerosF32(mc.experts, r.s);
        try m.owned.own(bias);
        return .{ .second = .{
            .router = router,
            .route = .{ .sigmoid_bias = .{ .expert_bias = bias, .norm = mc.norm, .scale = mc.scale } },
            .gate = reader.bank(banks[0], c.quant),
            .up = reader.bank(banks[1], c.quant),
            .down = reader.bank(banks[2], c.quant),
            .top_k = @intCast(mc.top_k),
            .experts = @intCast(mc.experts),
        } };
    }
    // transformers' Lfm2 spells the SwiGLU w1/w3/w2; mlx-lm's lfm2_moe converter gate/up/down.
    const w1 = r.has(try r.name(&nb, "layers.{d}.feed_forward.w1.weight", .{i}));
    const keys = if (w1) [3][]const u8{ "w1", "w3", "w2" } else [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var gate = try r.proj(try r.name(&nb, "layers.{d}.feed_forward.{s}", .{ i, keys[0] }), hidden);
    var up = try r.proj(try r.name(&nb, "layers.{d}.feed_forward.{s}", .{ i, keys[1] }), hidden);
    const down = try r.proj(try r.name(&nb, "layers.{d}.feed_forward.{s}", .{ i, keys[2] }), c.intermediate_size);
    const gu = if (gate.s.ctx != null) try reader.join(&m.owned, &.{ &gate, &up }, r.s) else b.linear.Joined{};
    for ([_]Linear{ gate, up, down }) |l| try m.owned.linear(l);
    return .{ .first = .{ .gate = gate, .up = up, .down = down, .gu = gu } };
}

/// The head as the engine finds it: `lm_head` beside the decoder, then at the root, then the tied embedding.
fn headOf(r: *reader.Reader, m: *Loaded, embed: Linear, hidden: u32, tied: bool) !Linear {
    var nb: [160]u8 = undefined;
    const nested = if (std.mem.eql(u8, r.prefix, "language_model.model")) "language_model" else r.prefix;
    for ([_][]const u8{ try std.fmt.bufPrint(&nb, "{s}.lm_head", .{nested}), "lm_head" }) |base| {
        var kb: [176]u8 = undefined;
        if (!r.has(try std.fmt.bufPrint(&kb, "{s}.weight", .{base}))) continue;
        const head = try r.table(base, hidden);
        try m.owned.linear(head);
        return head;
    }
    if (!tied) return error.MissingWeight;
    return embed;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

/// The tiny pack the conformance suite runs: a conv layer and an attention layer, width 64, seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    const inter = 128;
    var t = sdk.tensors.Tensors.init(a, 1);
    try t.rand("model.embed_tokens.weight", .{}, &.{ 64, h }, .F32, 0.5);
    try t.fill("model.embedding_norm.weight", .{}, &.{h}, .F32, 1);
    for (0..2) |i| {
        try t.fill("model.layers.{d}.operator_norm.weight", .{i}, &.{h}, .F32, 1);
        try t.fill("model.layers.{d}.ffn_norm.weight", .{i}, &.{h}, .F32, 1);
        try t.rand("model.layers.{d}.feed_forward.w1.weight", .{i}, &.{ inter, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.feed_forward.w3.weight", .{i}, &.{ inter, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.feed_forward.w2.weight", .{i}, &.{ h, inter }, .F32, 0.1);
        if (i == 0) {
            try t.rand("model.layers.{d}.conv.in_proj.weight", .{i}, &.{ 3 * h, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.conv.conv.weight", .{i}, &.{ h, 3, 1 }, .F32, 0.3);
            try t.rand("model.layers.{d}.conv.out_proj.weight", .{i}, &.{ h, h }, .F32, 0.1);
        } else {
            try t.rand("model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ 64, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.self_attn.out_proj.weight", .{i}, &.{ h, 64 }, .F32, 0.1);
            try t.fill("model.layers.{d}.self_attn.q_layernorm.weight", .{i}, &.{32}, .F32, 1);
            try t.fill("model.layers.{d}.self_attn.k_layernorm.weight", .{i}, &.{32}, .F32, 1);
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"lfm2","vocab_size":64,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"max_position_embeddings":4096,"norm_eps":1e-5,
        \\ "rope_theta":1000000.0,"conv_L_cache":3,"layer_types":["conv","full_attention"],"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
    _ = @import("../../sdk/config/json.zig");
}
