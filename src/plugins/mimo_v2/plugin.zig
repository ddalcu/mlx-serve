//! Xiaomi MiMo-V2 (`mimo_v2`, `mimo_v2_flash`): the decoder base over the engine's loaded tensors. MiMo's sink
//! attention (`sink_attention.zig`), a dense SwiGLU on the leading layers, then sigmoid-routed experts with a
//! selection-only bias, an f32 router and no shared expert, closing on add_norm seams. Batched rows decode too.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const gated = @import("../../sdk/bases/blocks/gated_attention.zig");
const reader = @import("../../sdk/bases/reader.zig");
const EitherMlp = @import("../lfm2/plugin.zig").EitherMlp;
const SinkAttention = @import("sink_attention.zig").SinkAttention;
const mtp_head = @import("mtp_head.zig");
const streamed = @import("../../sdk/weights/streamed.zig");
const sushi_pack = @import("../../sdk/quant/exl3/sushi_pack.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

pub const Mlp = EitherMlp(b.mlp.GatedMlp, moe.Moe);

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = SinkAttention,
    .mlp = Mlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
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
    .cache = &.{ .layout = &config.layout },
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded), .head = head_hooks },
    .weights = &.{ .resident_bytes = &residentBytes },
};

const head_hooks = reader.headHooks(Loaded, mtp_head.Heads, struct {
    /// The engine runs one head per draft depth (`SpecFacts.mtp_heads`): the pack must ship exactly those.
    fn load(ctx: *const abi.LoadCtx) !?*mtp_head.Heads {
        const h = (try mtp_head.load(ctx, config.of(ctx.parsed))) orelse return null;
        if (h.n == ctx.parsed.spec.mtp_heads) return h;
        h.deinit();
        gpa.destroy(h);
        return error.MimoMtpHeadCountDiffers;
    }
}.load, mtp_head.forward);

/// The per-layer geometry is the layout's; the engine bills from it.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A Sushi pack is held as its adapter leaves it (`SushiBill`); a streaming pack as its residents plus the stream's
/// rows (`slotRows`); every other pack as stored.
fn residentBytes(p: *const abi.Parsed, model_dir: abi.Str, settings_json: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    const c = config.of(p);
    out.* = shards;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (c.streamed) {
        const bank = streamed.Streamed.bankBytes(gpa, io, model_dir.slice(), slotRows(c, settings_json.slice())) catch |e| return Verdict.named(.failed, @errorName(e));
        out.* = shards +| bank;
        return .ok;
    }
    if (c.expert_block_json.len == 0) return .ok;
    out.* = sushi_pack.residentBytes(io, std.heap.page_allocator, model_dir.slice(), SushiBill{ .layers = c.num_hidden_layers }) catch |e|
        return Verdict.named(.failed, @errorName(e));
    return .ok;
}

/// A Sushi MiMo tensor as the load holds it: the towers and anything past the trunk not at all, a QKV part's tile
/// scales one row per code row, the dense MLP's dropped once it is decoded (bf16, twice its e4m3 codes).
pub const SushiBill = struct {
    layers: u32,

    pub fn bill(f: SushiBill, t: sushi_pack.Stored) !u64 {
        for ([_][]const u8{ "visual.", "audio_encoder.", "speech_embeddings." }) |u| if (std.mem.startsWith(u8, t.key, u)) return 0;
        if (sushi_pack.pastTrunk(t.key, f.layers)) return 0;
        if (std.mem.endsWith(u8, t.key, ".weight_scale_inv"))
            return if (std.mem.indexOf(u8, t.key, ".self_attn.qkv_proj.") != null) t.bytes * sushi_pack.FP8_BLOCK else 0;
        if (std.mem.eql(u8, t.dtype, "F8_E4M3") and std.mem.indexOf(u8, t.key, ".mlp.") != null) return 2 * t.bytes;
        return t.bytes;
    }
};

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},
    /// The sigmoid routing chain compiled once for every MoE layer (the engine's `compileMoeRouting`).
    route_fn: compiled.Closure = .{ .ctx = null },
    /// The MTP heads, once `HeadHooks.load` found them.
    head: ?*mtp_head.Heads = null,
    /// A streaming pack's routed experts (`experts.bin`).
    experts: ?streamed.Streamed = null,

    fn release(m: *Loaded) void {
        if (m.head) |h| {
            h.deinit();
            gpa.destroy(h);
        }
        if (m.experts) |*x| x.close();
        compiled.free(&m.route_fn);
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
    var r = try reader.Reader.open(ctx, c.block_json, "model");
    defer r.close();
    // A Sushi pack's routed experts are read and run by their own format (EXL3).
    var xr: ?reader.Reader = if (c.expert_block_json.len > 0) try reader.Reader.open(ctx, c.expert_block_json, "model") else null;
    defer if (xr) |*x| x.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s } };
    errdefer m.release();
    if (c.streamed) {
        const rows = try gpa.alloc(u32, c.num_hidden_layers - c.first_k_dense);
        defer gpa.free(rows);
        @memset(rows, slotRows(c, ctx.settings_json.slice()));
        m.experts = try streamed.Streamed.open(gpa, std.Io.Threaded.global_single_threaded.io(), ctx.model_dir.slice(), r.s, .{ .rows = rows });
    }
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(m, &r, if (xr) |*x| x else null, c, @intCast(i));
    // A streamed layer's decode calls predict the next routed layer's experts with its router.
    if (m.experts != null) for (m.layers[0 .. m.layers.len - 1], m.layers[1..]) |*L, *next| switch (L.mlp) {
        .second => |*w| if (next.mlp == .second and next.mlp.second.router.s.ctx == null) {
            w.streamed.?.next = .{ .router = next.mlp.second.router.w, .bias = next.mlp.second.route.sigmoid_bias.expert_bias };
        },
        .first => {},
    };

    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const embed = try r.table(try r.name(&nb, "embed_tokens", .{}), hidden);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try m.owned.own(final_norm);
    const head = if (r.has("lm_head.weight")) try r.table("lm_head", hidden) else if (c.tie_word_embeddings) embed else return error.MissingWeight;
    if (head.w.ctx != embed.w.ctx) try m.owned.linear(head);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = head,
    };
    return m;
}

/// A projection with its additive `.bias` and its decode side copy (sharing the bias).
fn proj(m: *Loaded, r: *reader.Reader, base: []const u8, in_dim: u32, layer: u32, layers: u32) !gated.Proj {
    var lin = try r.proj(base, in_dim);
    try m.owned.linear(lin);
    var nb: [160]u8 = undefined;
    lin.bias = try r.share(try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    try m.owned.own(lin.bias);
    var dq = try b.decode_copies.decodeCopy(lin, layer, layers, r.s);
    try m.owned.linear(dq);
    dq.bias = lin.bias;
    return .{ .lin = lin, .dq = dq };
}

fn loadLayer(m: *Loaded, r: *reader.Reader, xr: ?*reader.Reader, c: *const config.Config, i: u32) !Model.Layer {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const n = c.num_hidden_layers;
    var norms: [2]Array = undefined;
    for (&norms, [_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |*x, leaf| {
        x.* = try r.need(try r.name(&nb, "layers.{d}.{s}.weight", .{ i, leaf }));
        try m.owned.own(x.*);
    }
    var sinks = try r.share(try r.name(&nb, "layers.{d}.self_attn.sinks", .{i}));
    if (sinks.ctx == null) sinks = try r.share(try r.name(&nb, "layers.{d}.self_attn.attention_sink_bias", .{i}));
    try m.owned.own(sinks);
    const global = c.isGlobal(i);
    const mixer: SinkAttention.Weights = .{
        .q = try proj(m, r, try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden, i, n),
        .k = try proj(m, r, try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden, i, n),
        .v = try proj(m, r, try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden, i, n),
        .o = try proj(m, r, try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), c.num_attention_heads * c.v_head_dim, i, n),
        .sinks = sinks,
        .geo = .{
            .heads = @intCast(c.num_attention_heads),
            .kv_heads = @intCast(c.kvHeads(i)),
            .head_dim = @intCast(c.head_dim),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
            .rope_dims = @intCast(c.ropeDims()),
            .rope_base = if (global) c.rope_theta else c.swa_rope_theta,
            .rope_scale = 1.0,
        },
        .v_dim = @intCast(c.v_head_dim),
        .value_scale = c.value_scale,
        .window = if (global) 0 else c.sliding_window,
        .ring = if (global) 0 else c.ring(),
    };
    return .{ .input_norm = norms[0], .mixer = mixer, .post_norm = norms[1], .mlp = try mlpOf(m, r, xr, c, i) };
}

fn mlpOf(m: *Loaded, r: *reader.Reader, xr: ?*reader.Reader, c: *const config.Config, i: u32) !Mlp.Weights {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    if (!c.moeAt(i)) {
        var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
        var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
        const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
        const gu = if (gate.s.ctx != null) try reader.join(&m.owned, &.{ &gate, &up }, r.s) else b.linear.Joined{};
        for ([_]Linear{ gate, up, down }) |l| try m.owned.linear(l);
        return .{ .first = .{ .gate = gate, .up = up, .down = down, .gu = gu } };
    }
    // The router is `mlp.gate`, or `mlp.gate.proj` on HF-native builds.
    const proj_router = !r.has(try r.name(&nb, "layers.{d}.mlp.gate.weight", .{i})) and r.has(try r.name(&nb, "layers.{d}.mlp.gate.proj.weight", .{i}));
    var router = try r.proj(if (proj_router) try r.name(&nb, "layers.{d}.mlp.gate.proj", .{i}) else try r.name(&nb, "layers.{d}.mlp.gate", .{i}), hidden);
    try m.owned.linear(router);
    // MiMo routes in f32 (the reference's `F.linear` on f32 inputs and weights): a dense router widens once.
    if (router.s.ctx == null and mlx.mlx_array_dtype(router.w) != .float32) {
        var f = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, router.w, .float32, r.s));
        try mlx.check(mlx.mlx_array_eval(f));
        try m.owned.own(f);
        router.w = f;
    }
    var banks: [3]moe.Bank = @splat(.{ .w = .{}, .qp = .{ .bits = 0, .group_size = 0 } });
    var fb: ?moe.FormatBanks = null;
    if (xr) |x| {
        var fw: [3]@import("../../abi/quant.zig").Weight = undefined;
        for (&fw, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, c.moe_intermediate }) |*w, leaf, in_dim| {
            w.* = try x.formatBank(try x.name(&nb, "layers.{d}.mlp.switch_mlp.{s}", .{ i, leaf }), in_dim);
            for ([_]@import("../../abi/quant.zig").Array{ w.w, w.scales, w.biases }) |a| try m.owned.own(.{ .ctx = a.ctx });
        }
        fb = .{ .table = x.fmt, .gate = fw[0], .up = fw[1], .down = fw[2] };
    } else if (m.experts != null) {
        // The bank's quant params only: each call's fetch names the slots.
        if (c.quant.mode != .mxfp4) return error.MimoStreamedBankNotMxfp4;
        banks = @splat(reader.bank(.{}, c.quant));
    } else for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, c.moe_intermediate }) |*bk, leaf, in_dim| {
        const l = try r.proj(try r.name(&nb, "layers.{d}.mlp.switch_mlp.{s}", .{ i, leaf }), in_dim);
        try m.owned.linear(l);
        bk.* = reader.bank(l, c.quant);
    }
    // Some conversions drop the selection bias; the reference zero-inits it (plain sigmoid top-k).
    var bias = try r.share(try r.name(&nb, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}));
    if (bias.ctx == null) bias = try reader.zerosF32(c.experts, r.s);
    try m.owned.own(bias);
    var w: moe.Moe.Weights = .{
        .router = router,
        .route = .{ .sigmoid_bias = .{ .expert_bias = bias, .norm = c.route_norm, .scale = c.route_scale } },
        .gate = banks[0],
        .up = banks[1],
        .down = banks[2],
        .top_k = @intCast(c.top_k),
        .experts = @intCast(c.experts),
        .format_banks = fb,
    };
    if (m.experts) |*x| w.streamed = .{ .routed = x.routedSource(), .layer = i };
    if (m.route_fn.ctx == null) m.route_fn = moe.compileRouting(&w, r.s);
    w.route_fn = m.route_fn;
    return .{ .second = w };
}

/// A streaming pack's slot rows per routed layer: the model's `stream_experts`, else every expert.
fn slotRows(c: *const config.Config, settings_json: []const u8) u32 {
    return if (streamed.Setting.parse(settings_json)) |st| st.slotRows(c.experts) else c.experts;
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

/// The tiny pack the conformance suite runs: a dense global layer, an MoE sliding layer with sinks, width 64,
/// head 32 (V 16, RoPE on 16 channels), seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    var t = sdk.tensors.Tensors.init(a, 1);
    try t.rand("model.embed_tokens.weight", .{}, &.{ 64, h }, .F32, 0.5);
    try t.fill("model.norm.weight", .{}, &.{h}, .F32, 1);
    try t.rand("lm_head.weight", .{}, &.{ 64, h }, .F32, 0.1);
    for (0..2) |i| {
        try t.fill("model.layers.{d}.input_layernorm.weight", .{i}, &.{h}, .F32, 1);
        try t.fill("model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}, .F32, 1);
        const kvh: u64 = if (i == 0) 1 else 2;
        try t.rand("model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ 64, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ kvh * 32, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ kvh * 16, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, 32 }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.q_proj.bias", .{i}, &.{64}, .F32, 0.1);
        if (i == 0) {
            try t.rand("model.layers.{d}.mlp.gate_proj.weight", .{i}, &.{ 128, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.up_proj.weight", .{i}, &.{ 128, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, 128 }, .F32, 0.1);
        } else {
            try t.rand("model.layers.{d}.self_attn.attention_sink_bias", .{i}, &.{2}, .F32, 0.5);
            try t.rand("model.layers.{d}.mlp.gate.weight", .{i}, &.{ 4, h }, .F32, 0.3);
            try t.rand("model.layers.{d}.mlp.gate.e_score_correction_bias", .{i}, &.{4}, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.gate_proj.weight", .{i}, &.{ 4, 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.up_proj.weight", .{i}, &.{ 4, 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.down_proj.weight", .{i}, &.{ 4, h, 32 }, .F32, 0.1);
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"mimo_v2","vocab_size":64,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"swa_num_key_value_heads":2,"head_dim":32,"v_head_dim":16,
        \\ "layernorm_epsilon":1e-6,"rope_theta":10000000.0,"swa_rope_theta":10000.0,
        \\ "rope_parameters":{"partial_rotary_factor":0.5,"rope_theta":10000000.0},"sliding_window":8,
        \\ "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],"max_position_embeddings":4096,"n_routed_experts":4,
        \\ "num_experts_per_tok":2,"moe_intermediate_size":32,"attention_value_scale":0.707,"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
}
