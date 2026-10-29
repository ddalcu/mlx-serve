//! Muse-Glimmer-30B's text trunk (dense GQA 32/2, hd 128): `Decoder(NormedEmbed, RmsNorm,
//! PostNormed(DecoderAttention(.sliding, .gate = .channel)), PostNormed(SwiGlu), ResidualTail, SoftcapHead)`:
//! RMS-normed embeddings, Gemma2-centred sandwich norms (1+w folded at load, post-norms at `post_norm_eps`) under a
//! plain final norm, a weight-less q/k norm with `qk_scale_factor` folded into the scale, a per-channel sigmoid output
//! gate, NoPE full layers, sliding layers, logits times `output_multiplier` then softcapped. `vision.zig` is the
//! tower the engine's `VisionEncoder` runs beside the plugin.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const quant_abi = @import("../../abi/quant.zig");
const bases = @import("../../sdk/bases/bases.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const loader = @import("../llama/load.zig");
const NormedEmbed = @import("embed.zig").NormedEmbed;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Model = bases.Decoder(.{
    .embed = NormedEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.extras.PostNormed(b.decoder_attention.DecoderAttention(.{ .sliding = true, .gate = .channel })),
    .mlp = b.extras.PostNormed(b.mlp.SwiGlu),
    .seam = b.seam.ResidualTail,
    .head = b.extras.SoftcapHead,
});

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{},
    .weights = &.{},
    .spec = &.{ .trunk_weight = &trunkWeight },
};

const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;

/// Every term is the engine's default arithmetic over the shell.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: loader.Owned = .{},
    softcap: compiled.Closure = .{ .ctx = null },

    fn release(m: *Loaded) void {
        compiled.free(&m.softcap);
        m.owned.release();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

/// `1 + w` as the legacy folds a Gemma2-centred norm (bf16 one + the weight), owned by `o`; frees `raw`.
fn addOne(o: *loader.Owned, raw: Array, s: mlx.mlx_stream) !Array {
    defer _ = mlx.mlx_array_free(raw);
    const f = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(f);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_astype(&one, f, .bfloat16, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, one, raw, s));
    try o.own(out);
    return out;
}

/// bf16 ones `[n]`, owned by `o`.
fn ones(o: *loader.Owned, n: u32, s: mlx.mlx_stream) !Array {
    const one = try loader.bf16Scalar(o, 1.0, s);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_full(&out, &[_]c_int{@intCast(n)}, 1, one, .bfloat16, s));
    try o.own(out);
    return out;
}

const Softcap = struct { cap: Array, s: mlx.mlx_stream };

/// `tanh(x / cap) * cap`, the chain the legacy compiles (shapeless) for every softcapped model.
fn softcapBody(p: *const Softcap, in: []const Array, out: *[1]Array) anyerror!void {
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_divide(&scaled, in[0], p.cap, p.s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_tanh(&t, scaled, p.s));
    out[0] = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out[0], t, p.cap, p.s));
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{ "model.language_model", "model" }, "embed_tokens.weight");
    defer r.close();
    const s = r.s;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.post_norm_eps), .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    const o = &m.owned;

    const hidden = c.hidden_size;
    const qk_ones = try ones(o, c.head_dim, s);
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
        var k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
        var v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
        var gate = try r.proj(try r.name(&nb, "layers.{d}.self_attn.gate_proj", .{i}), hidden);
        const qkv = if (q.s.ctx != null) try loader.join(o, &.{ &q, &k, &v, &gate }, s) else b.linear.Joined{};
        for ([_]Linear{ q, k, v, gate }) |l| try o.linear(l);
        const ow = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), c.num_attention_heads * c.head_dim);
        try o.linear(ow);
        var mg = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
        var mu = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
        const md = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
        const gu = if (mg.s.ctx != null) try loader.join(o, &.{ &mg, &mu }, s) else b.linear.Joined{};
        for ([_]Linear{ mg, mu, md }) |l| try o.linear(l);

        const global = c.global[i];
        L.* = .{
            .input_norm = try addOne(o, try r.need(try r.name(&nb, "layers.{d}.input_layernorm.weight", .{i})), s),
            .mixer = .{
                .inner = .{
                    .q = q,
                    .k = k,
                    .v = v,
                    .o = ow,
                    .q_norm = qk_ones,
                    .k_norm = qk_ones,
                    .qkv = qkv,
                    .out_gate = gate,
                    .geo = .{
                        .heads = @intCast(c.num_attention_heads),
                        .kv_heads = @intCast(c.num_key_value_heads),
                        .head_dim = @intCast(c.head_dim),
                        .scale = c.qk_scale_factor / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
                        .rope_dims = @intCast(c.head_dim),
                        .rope_base = if (global) c.rope_theta else c.rope_local_base_freq,
                        .rope_scale = 1.0,
                        .nope = c.no_rope[i],
                    },
                    .window = if (global) 0 else c.sliding_window,
                },
                .norm = try addOne(o, try r.need(try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{i})), s),
            },
            .post_norm = try addOne(o, try r.need(try r.name(&nb, "layers.{d}.pre_feedforward_layernorm.weight", .{i})), s),
            .mlp = .{
                .inner = .{ .gate = mg, .up = mu, .down = md, .gu = gu, .rows_joined = qkv.built() },
                .norm = try addOne(o, try r.need(try r.name(&nb, "layers.{d}.post_feedforward_layernorm.weight", .{i})), s),
            },
        };
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), hidden);
    try o.linear(embed);
    // The final norm is plain-scale (ones-init), unlike the layers' centred ones.
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try o.own(final_norm);
    const cap: Array = if (c.softcap > 0) try loader.bf16Scalar(o, c.softcap, s) else .{ .ctx = null };
    if (cap.ctx != null) m.softcap = compiled.compile(Softcap, .{ .cap = cap, .s = s }, 1, softcapBody, true);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16, .ones = try ones(o, hidden, s) },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = .{
            .head = try loader.headOf(&r, o, embed, hidden, c.tie_word_embeddings),
            .cap = cap,
            .mult = try loader.bf16Scalar(o, c.output_multiplier, s),
            .compiled = m.softcap,
        },
    };
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

/// The bare table and head (no norm, multiplier or softcap) a DFlash drafter reads, as the legacy `rawEmbedding` and
/// `lmHeadForDraft`; borrowed until unload.
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    b.trunk_weight.trunkWeight(switch (which) {
        .embed => &m.base.weights.embed.table,
        .head => &m.base.weights.head.head,
    }, out);
    return .ok;
}

/// The tiny pack the conformance suite runs: the flat text sibling, two layers (a sliding one, a NoPE full one), width
/// 64, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = sdk.fixture;
    const V = 64;
    const H = 64;
    const HD = 32;
    const HEADS = 2;
    const INTER = 128;
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 23;
    const zeros = try a.alloc(f32, H);
    @memset(zeros, 0);
    const Put = struct {
        a: std.mem.Allocator,
        list: *std.ArrayList(fx.Tensor),
        seed: *u64,
        fn rand(p: @This(), name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            p.seed.* += 1;
            try p.list.append(p.a, .{ .name = name, .shape = try p.a.dupe(u64, shape), .data = try fx.randomF32(p.a, p.seed.*, n, scale) });
        }
        fn at(p: @This(), li: usize, leaf: []const u8) ![]const u8 {
            return std.fmt.allocPrint(p.a, "model.layers.{d}.{s}", .{ li, leaf });
        }
    };
    const p: Put = .{ .a = a, .list = &tensors, .seed = &seed };
    try p.rand("model.embed_tokens.weight", &.{ V, H }, 0.5);
    try p.rand("lm_head.weight", &.{ V, H }, 0.5);
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{H}, .data = zeros });
    for (0..2) |i| {
        // Centred norms: a zero weight is a unit scale.
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try tensors.append(a, .{ .name = try p.at(i, nm), .shape = &.{H}, .data = zeros });
        try p.rand(try p.at(i, "self_attn.q_proj.weight"), &.{ HEADS * HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.k_proj.weight"), &.{ HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.v_proj.weight"), &.{ HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.gate_proj.weight"), &.{ HEADS * HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.o_proj.weight"), &.{ H, HEADS * HD }, 0.1);
        try p.rand(try p.at(i, "mlp.gate_proj.weight"), &.{ INTER, H }, 0.1);
        try p.rand(try p.at(i, "mlp.up_proj.weight"), &.{ INTER, H }, 0.1);
        try p.rand(try p.at(i, "mlp.down_proj.weight"), &.{ H, INTER }, 0.1);
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"muse_glimmer_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
        \\"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"rms_norm_eps":1e-5,"post_norm_eps":1e-8,
        \\"final_logit_softcapping":20.0,"vocab_size":64,"max_position_embeddings":4096,"tie_word_embeddings":false,
        \\"eos_token_id":2,"sliding_window":4,"layer_types":["sliding_attention","full_attention"],
        \\"layer_rope_theta":[500000.0,0],"rope_parameters":{"rope_theta":500000.0,"rope_type":"default"}}
        },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
