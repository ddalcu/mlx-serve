//! LFM3 (`lfm3`): `Decoder(TableEmbed, RmsNorm, PostNormed(Either(GatedConv, Attention)), PostNormed(GatedMlp),
//! ResidualTail, LmHead)` from `sdk/bases`. Each layer is a width-7 gated short conv or q/k-normed attention (full, or
//! sliding over `sliding_window` keys) with a per-head sigmoid output gate; the operator and the SwiGLU each close on
//! their own post norm (the sandwich). Full layers rotate only their first pairs (the zero tail). mlx-vlm's `lfm3`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const reader = @import("../../sdk/bases/reader.zig");
pub const config = @import("config.zig");
const vision = @import("vision.zig");
const media = @import("media.zig");
const audio = @import("audio.zig");
pub const depth = @import("depth.zig");
pub const codec = @import("codec.zig");
const speech = @import("speech.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const extras = b.extras;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

pub const Attention = b.decoder_attention.DecoderAttention(.{ .sliding = true, .gate = .head });
pub const Mixer = block.Either(b.gated_conv.GatedConv, Attention);

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = extras.PostNormed(Mixer),
    .mlp = extras.PostNormed(b.mlp.GatedMlp),
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

comptime {
    if (Model.batches_decode != config.batches_decode) @compileError("lfm3: config.batches_decode disagrees with the base");
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
    .media = &.{ .tower = &media.tower, .prepare = &media.prepare, .placeholder_rows = &media.placeholderRows },
    .speech = &speech.hooks,
};

/// Every term is the engine's default arithmetic over the shell and the layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},
    /// LFM3-VL's tower and projector; null on a text pack (or one shipped without them).
    tower: ?vision.Tower = null,
    /// The speech encoder and projector (lfm3_audio, Omni).
    speech: ?audio.Tower = null,
    /// Speech output: the depth decoder and the frames' input table.
    depth: ?depth.Depth = null,
    /// The speech codec beside the pack (`audio_tokenizer/`).
    codec: ?codec.Codec = null,

    fn release(m: *Loaded) void {
        if (m.codec) |*cd| cd.deinit();
        if (m.depth) |*d| d.deinit();
        if (m.tower) |*t| t.deinit();
        if (m.speech) |*t| t.deinit();
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
    const prefix = try reader.probePrefix(ctx, "language_model.model", &.{ "model.language_model", "model" }, "embed_tokens.weight");
    var r = try reader.Reader.open(ctx, c.block_json, prefix);
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.norm_eps, c.norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    // One zero-tail table serves every full layer.
    const full_freqs = try zeroTailWavelengths(c.full_rope, c.head_dim);
    if (full_freqs.ctx != null) try m.owned.own(full_freqs);
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(m, &r, c, @intCast(i), full_freqs);

    var nb: [256]u8 = undefined;
    const embed = try r.table(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "embedding_norm.weight", .{}));
    try m.owned.own(final_norm);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = try headOf(&r, m, embed, c.hidden_size, c.tie_word_embeddings),
    };
    if (c.vision) |v| m.tower = try vision.Tower.load(ctx, v, c.block_json, &m.owned);
    if (c.audio) |a| m.speech = try audio.Tower.load(ctx, a, c.block_json, &m.owned);
    if (c.speech != null) m.codec = try codec.Codec.load(ctx, &m.owned);
    if (c.speech) |*sp| m.depth = try depth.Depth.load(ctx, c.block_json, &m.owned, sp.codebooks, sp.eos_code, sp.maskgit(), sp.depth_eps, sp.depth_theta, sp.depth_attention[0..sp.depth_layers], c.hidden_size);
    return m;
}

fn norm(m: *Loaded, r: *reader.Reader, comptime fmt: []const u8, i: u32) !Array {
    var nb: [256]u8 = undefined;
    const w = try r.need(try r.name(&nb, fmt, .{i}));
    try m.owned.own(w);
    return w;
}

fn proj(r: *reader.Reader, comptime fmt: []const u8, i: u32, in_dim: u32) !Linear {
    var nb: [256]u8 = undefined;
    return r.proj(try r.name(&nb, fmt, .{i}), in_dim);
}

fn loadLayer(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32, full_freqs: Array) !Model.Layer {
    const hidden = c.hidden_size;
    const op: Mixer.Weights = switch (c.layers[i]) {
        .conv => blk: {
            const in_proj = try proj(r, "layers.{d}.conv.in_proj", i, hidden);
            try m.owned.linear(in_proj);
            var nb: [256]u8 = undefined;
            const conv = try reader.depthwiseConv(try r.need(try r.name(&nb, "layers.{d}.conv.conv.weight", .{i})), r.s);
            try m.owned.own(conv);
            const out_proj = try proj(r, "layers.{d}.conv.out_proj", i, hidden);
            try m.owned.linear(out_proj);
            break :blk .{ .first = .{ .in_proj = in_proj, .conv = conv, .out_proj = out_proj, .hidden = @intCast(hidden), .kernel = @intCast(c.conv_kernel) } };
        },
        .full, .sliding => |kind| blk: {
            var q = try proj(r, "layers.{d}.self_attn.q_proj", i, hidden);
            var k = try proj(r, "layers.{d}.self_attn.k_proj", i, hidden);
            var v = try proj(r, "layers.{d}.self_attn.v_proj", i, hidden);
            var gate = try proj(r, "layers.{d}.self_attn.gate_proj", i, hidden);
            const o = try proj(r, "layers.{d}.self_attn.o_proj", i, c.num_attention_heads * c.head_dim);
            const qkv = if (q.s.ctx != null) try reader.join(&m.owned, &.{ &q, &k, &v, &gate }, r.s) else b.linear.Joined{};
            for ([_]Linear{ q, k, v, gate, o }) |l| try m.owned.linear(l);
            const rope = if (kind == .full) c.full_rope else c.sliding_rope;
            break :blk .{ .second = .{
                .q = q,
                .k = k,
                .v = v,
                .o = o,
                .out_gate = gate,
                .q_norm = try norm(m, r, "layers.{d}.self_attn.q_norm.weight", i),
                .k_norm = try norm(m, r, "layers.{d}.self_attn.k_norm.weight", i),
                .qkv = qkv,
                .window = if (kind == .sliding) c.sliding_window else 0,
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
                    .rope_dims = @intCast(c.head_dim),
                    .rope_base = rope.theta,
                    .rope_scale = 1.0,
                    .freqs = if (rope.angles * 2 < c.head_dim) full_freqs else .{},
                },
            } };
        },
    };
    return .{
        .input_norm = try norm(m, r, "layers.{d}.operator_norm.weight", i),
        .mixer = .{ .inner = op, .norm = try norm(m, r, "layers.{d}.post_operator_norm.weight", i) },
        .post_norm = try norm(m, r, "layers.{d}.ffn_norm.weight", i),
        .mlp = .{ .inner = try mlpOf(m, r, c, i), .norm = try norm(m, r, "layers.{d}.post_ffn_norm.weight", i) },
    };
}

fn mlpOf(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !b.mlp.GatedMlp.Weights {
    var gate = try proj(r, "layers.{d}.feed_forward.w1", i, c.hidden_size);
    var up = try proj(r, "layers.{d}.feed_forward.w3", i, c.hidden_size);
    const down = try proj(r, "layers.{d}.feed_forward.w2", i, c.intermediate_size);
    const gu = if (gate.s.ctx != null) try reader.join(&m.owned, &.{ &gate, &up }, r.s) else b.linear.Joined{};
    for ([_]Linear{ gate, up, down }) |l| try m.owned.linear(l);
    return .{ .gate = gate, .up = up, .down = down, .gu = gu };
}

/// The head: `lm_head` beside the decoder, then at the root, then the tied embedding.
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

/// MLX rope's wavelengths (`angle = position / freqs[i]`) for a head whose first `rope.angles` pairs turn at the full
/// head's progression (`theta^(2i / head_dim)`) and whose tail never turns (an infinite wavelength); empty when every
/// pair turns (the plain `rope_base` serves it).
pub fn zeroTailWavelengths(rope: config.Rope, head_dim: u32) !Array {
    const pairs = head_dim / 2;
    if (rope.angles >= pairs) return .{};
    var buf: [512]f32 = undefined;
    if (pairs > buf.len) return error.HeadTooWide;
    for (buf[0..pairs], 0..) |*f, i| f.* = if (i < rope.angles)
        @floatCast(std.math.pow(f64, rope.theta, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim))))
    else
        std.math.inf(f32);
    const a = mlx.mlx_array_new_data(&buf, &[_]c_int{@intCast(pairs)}, 1, .float32);
    if (a.ctx == null) return error.OutOfMemory;
    return a;
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

/// The tiny pack the conformance suite runs: a conv, a full and a sliding layer, width 64, seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    const inter = 96;
    const hd = 16;
    var t = sdk.tensors.Tensors.init(a, 3);
    try t.rand("language_model.model.embed_tokens.weight", .{}, &.{ 64, h }, .F32, 0.5);
    try t.fill("language_model.model.embedding_norm.weight", .{}, &.{h}, .F32, 1);
    for (0..3) |i| {
        for ([_][]const u8{ "operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm" }) |n|
            try t.fill("language_model.model.layers.{d}.{s}.weight", .{ i, n }, &.{h}, .F32, 1);
        try t.rand("language_model.model.layers.{d}.feed_forward.w1.weight", .{i}, &.{ inter, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.feed_forward.w3.weight", .{i}, &.{ inter, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.feed_forward.w2.weight", .{i}, &.{ h, inter }, .F32, 0.1);
        if (i == 0) {
            try t.rand("language_model.model.layers.{d}.conv.in_proj.weight", .{i}, &.{ 3 * h, h }, .F32, 0.1);
            try t.rand("language_model.model.layers.{d}.conv.conv.weight", .{i}, &.{ h, 7, 1 }, .F32, 0.3);
            try t.rand("language_model.model.layers.{d}.conv.out_proj.weight", .{i}, &.{ h, h }, .F32, 0.1);
            continue;
        }
        try t.rand("language_model.model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ 4 * hd, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ 2 * hd, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ 2 * hd, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.self_attn.gate_proj.weight", .{i}, &.{ 4, h }, .F32, 0.1);
        try t.rand("language_model.model.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, 4 * hd }, .F32, 0.1);
        try t.fill("language_model.model.layers.{d}.self_attn.q_norm.weight", .{i}, &.{hd}, .F32, 1);
        try t.fill("language_model.model.layers.{d}.self_attn.k_norm.weight", .{i}, &.{hd}, .F32, 1);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"lfm3","vocab_size":64,"hidden_size":64,"intermediate_size":96,"num_hidden_layers":3,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"max_position_embeddings":4096,"norm_eps":1e-5,
        \\ "conv_L_cache":7,"sliding_window":4,"layer_types":["conv","full_attention","sliding_attention"],"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
    _ = media;
    _ = audio;
    _ = depth;
    _ = codec;
}
