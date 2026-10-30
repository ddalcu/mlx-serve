//! Thinking Machines Inkling (`inkling_mm_model`, text): `Decoder(TableEmbed, RmsNorm, Attention, Mlp, AddNorm, Head)`
//! with the embeddings RMS-normed and the final hidden divided by the muP width multiplier (`Hooks`). Attention has
//! no RoPE (a hidden-conditioned relative bias, short convs on k/v, log-scaling on global layers past the floor); each
//! block's output passes a short conv, whose windows are the slot's per-layer `.conv` state; dense bottom layers carry
//! a learned output gain, the rest route over experts plus shared experts in one softmax; the head keeps the padded
//! vocab's real rows. Serial, as the legacy.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const loader = @import("../llama/load.zig");
const attention = @import("attention.zig");
const moe = @import("moe.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const gpa = std.heap.c_allocator;

/// The embedding norm (null = none) and the head's input divisor.
pub const Hooks = struct {
    pub const Extra = struct { embed_norm: Array = .{}, mup: f32 = 1.0 };

    pub fn afterEmbed(x: *const Extra, c: *block.Ctx, h: ops.Array) !ops.Array {
        return if (x.embed_norm.ctx != null) c.g.rmsNorm(h, x.embed_norm, c.shared.eps) else h;
    }

    /// `hidden × (1/mup)` by an f32 0-d array, as the legacy multiplies.
    pub fn beforeHead(x: *const Extra, c: *block.Ctx, h: ops.Array) !ops.Array {
        if (x.mup == 1.0) return h;
        return c.g.mul(h, try c.g.adopt(mlx.mlx_array_new_float(1.0 / x.mup)));
    }
};

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = attention.Attention,
    .mlp = attention.Mlp,
    .seam = b.seam.AddNorm,
    .head = attention.Head,
    .hooks = Hooks,
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
    .cache = &.{ .layout = config.layout },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &trunkWeight },
};

/// Every term is the engine's default arithmetic over the shell and the layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: loader.Owned = .{},

    fn release(m: *Loaded) void {
        m.owned.release();
        gpa.free(m.layers);
        m.base.shared.deinit();
    }
};

/// The bare `embed` and `unembed` tables a drafter reads (no embedding norm, muP divisor or vocab trim).
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *@import("../../abi/quant.zig").Weight) callconv(.c) abi.Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    b.trunk_weight.trunkWeight(switch (which) {
        .embed => &m.base.weights.embed.table,
        .head => &m.base.weights.head.lin,
    }, out);
    return .ok;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    if (f.mode == .batched_rows) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{"model.llm"}, "embed.weight");
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(&r, &m.owned, c, @intCast(i));

    var nb: [256]u8 = undefined;
    const o = &m.owned;
    const hidden = c.hidden_size;
    const embed = try r.tableAt(try r.name(&nb, "embed", .{}), hidden);
    try o.linear(embed);
    const unembed = try r.tableAt(try r.name(&nb, "unembed", .{}), hidden);
    try o.linear(unembed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try o.own(final_norm);
    var extra: Hooks.Extra = .{ .mup = c.mup };
    if (c.embed_norm) {
        extra.embed_norm = try r.share(try r.name(&nb, "embed_norm.weight", .{}));
        if (extra.embed_norm.ctx == null) extra.embed_norm = try r.share(try r.name(&nb, "embedding_norm.weight", .{}));
        if (extra.embed_norm.ctx != null) try o.own(extra.embed_norm);
    }
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = .{ .lin = unembed, .vocab = @intCast(c.unpadded_vocab_size) },
        .extra = extra,
    };
    return m;
}

fn loadLayer(r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: u32) !Model.Layer {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const s = r.s;
    const heads_dim = c.num_attention_heads * c.head_dim;
    var lins: [5]Linear = undefined;
    for (&lins, [_][]const u8{ "wq_du", "wk_dv", "wv_dv", "wr_du", "wo_ud" }, 0..) |*l, nm, j| {
        l.* = try r.proj(try r.name(&nb, "layers.{d}.attn.{s}", .{ i, nm }), if (j == 4) heads_dim else hidden);
        try o.linear(l.*);
    }
    var dqs: [5]Linear = undefined;
    for (&dqs, lins) |*d, l| {
        d.* = try b.decode_copies.decodeCopy(l, i, c.num_hidden_layers, s);
        try o.linear(d.*);
    }
    const need = struct {
        fn f(rd: *loader.Reader, ow: *loader.Owned, layer: u32, comptime leaf: []const u8) !Array {
            var buf: [256]u8 = undefined;
            const a = try rd.need(try rd.name(&buf, "layers.{d}." ++ leaf, .{layer}));
            try ow.own(a);
            return a;
        }
    }.f;
    const global = c.global[i];
    const attn: attention.Attention.Weights = .{
        .q = lins[0],
        .k = lins[1],
        .v = lins[2],
        .rel = lins[3],
        .o = lins[4],
        .q_norm = try need(r, o, i, "attn.q_norm.weight"),
        .k_norm = try need(r, o, i, "attn.k_norm.weight"),
        .rel_proj = try need(r, o, i, "attn.rel_logits_proj.proj"),
        .k_sconv = try need(r, o, i, "attn.k_sconv.weight"),
        .v_sconv = try need(r, o, i, "attn.v_sconv.weight"),
        .out_sconv = try need(r, o, i, "attn_sconv.weight"),
        .heads = @intCast(c.num_attention_heads),
        .kv_heads = @intCast(c.num_key_value_heads),
        .head_dim = @intCast(c.head_dim),
        .d_rel = @intCast(c.d_rel),
        .window = if (global) 0 else c.sliding_window,
        .log_floor = c.log_n_floor,
        .log_alpha = c.log_alpha,
        .dq = dqs,
    };
    return .{
        .input_norm = try need(r, o, i, "attn_norm.weight"),
        .mixer = attn,
        .post_norm = try need(r, o, i, "mlp_norm.weight"),
        .mlp = .{
            .body = if (i < c.first_moe) .{ .dense = try denseOf(r, o, c, i) } else .{ .moe = try moeOf(r, o, c, i) },
            .out_sconv = try need(r, o, i, "mlp_sconv.weight"),
            .state_offset = @intCast(c.mlpStateOffset()),
        },
    };
}

fn denseOf(r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: u32) !@FieldType(@FieldType(attention.Mlp.Weights, "body"), "dense") {
    var nb: [256]u8 = undefined;
    var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), c.hidden_size);
    var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), c.hidden_size);
    const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
    const gu = if (gate.s.ctx != null) try loader.join(o, &.{ &gate, &up }, r.s) else b.linear.Joined{};
    for ([_]Linear{ gate, up, down }) |l| try o.linear(l);
    const gain = try r.share(try r.name(&nb, "layers.{d}.mlp.global_scale", .{i}));
    if (gain.ctx != null) try o.own(gain);
    return .{ .mlp = .{ .gate = gate, .up = up, .down = down, .gu = gu }, .gain = gain };
}

/// The router `mlp.gate` (dense, widened to f32, the shared experts' rows last), its selection bias `mlp.gate.bias`
/// (zeros where the pack has none) and output scale `mlp.gate.global_scale`, the stacked `mlp.experts.*` and
/// `mlp.shared_experts.*` banks.
fn moeOf(r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: u32) !moe.Moe.Weights {
    var nb: [256]u8 = undefined;
    const s = r.s;
    const router = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate", .{i}), c.hidden_size);
    try o.linear(router);
    if (router.s.ctx != null) return error.QuantizedInklingRouter;
    var r32 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r32);
    try mlx.check(mlx.mlx_astype(&r32, router.w, .float32, s));
    try mlx.check(mlx.mlx_array_eval(r32));
    try o.own(r32);

    var bias = try r.share(try r.name(&nb, "layers.{d}.mlp.gate.bias", .{i}));
    if (bias.ctx == null) try mlx.check(mlx.mlx_zeros(&bias, &[_]c_int{@intCast(c.num_experts)}, 1, .float32, s));
    try o.own(bias);

    var scale = c.route_scale;
    const gs = try r.share(try r.name(&nb, "layers.{d}.mlp.gate.global_scale", .{i}));
    if (gs.ctx != null) {
        defer _ = mlx.mlx_array_free(gs);
        var g32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g32);
        try mlx.check(mlx.mlx_astype(&g32, gs, .float32, s));
        try mlx.check(mlx.mlx_array_eval(g32));
        scale *= (mlx.mlx_array_data_float32(g32) orelse return error.NoData)[0];
    }

    var w: moe.Moe.Weights = .{
        .router = r32,
        .bias = bias,
        .routed = undefined,
        .shared = undefined,
        .top_k = @intCast(c.top_k),
        .n_shared = @intCast(c.n_shared),
        .scale = scale,
    };
    for ([_][]const u8{ "mlp.experts", "mlp.shared_experts" }, [_]*[3]b.moe.Bank{ &w.routed, &w.shared }) |container, banks| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, banks, 0..) |nm, *bk, j| {
            bk.* = try r.bank(try r.name(&nb, "layers.{d}.{s}.{s}", .{ i, container, nm }), if (j == 2) c.moe_intermediate_size else c.hidden_size);
            for ([_]Array{ bk.w, bk.s, bk.b }) |a| try o.own(a);
        }
    }
    return w;
}

/// The tiny pack the conformance suite runs: a dense sliding layer and a routed global one at width 64 (2 heads of 32,
/// one KV head, 4 experts of 32 plus one shared, top 2), 60 of 64 vocab rows real; seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = sdk.fixture;
    const V = 64;
    const H = 64;
    const HD = 32;
    const E = 4;
    const NS = 1;
    const MI = 32;
    const INTER = 128;
    const K = 4;
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 41;
    const Put = struct {
        a: std.mem.Allocator,
        list: *std.ArrayList(fx.Tensor),
        seed: *u64,
        fn rand(p: @This(), name: []const u8, shape: []const u64, scale: f32, offset: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            p.seed.* += 1;
            const data = try fx.randomF32(p.a, p.seed.*, n, scale);
            for (data) |*x| x.* += offset;
            try p.list.append(p.a, .{ .name = try std.fmt.allocPrint(p.a, "model.llm.{s}", .{name}), .shape = try p.a.dupe(u64, shape), .data = data });
        }
        fn at(p: @This(), li: usize, leaf: []const u8) ![]const u8 {
            return std.fmt.allocPrint(p.a, "layers.{d}.{s}", .{ li, leaf });
        }
    };
    const p: Put = .{ .a = a, .list = &tensors, .seed = &seed };
    try p.rand("embed.weight", &.{ V, H }, 0.5, 0);
    try p.rand("unembed.weight", &.{ V, H }, 0.5, 0);
    try p.rand("embed_norm.weight", &.{H}, 0.1, 1);
    try p.rand("norm.weight", &.{H}, 0.1, 1);
    for (0..2) |i| {
        try p.rand(try p.at(i, "attn_norm.weight"), &.{H}, 0.1, 1);
        try p.rand(try p.at(i, "mlp_norm.weight"), &.{H}, 0.1, 1);
        try p.rand(try p.at(i, "attn.q_norm.weight"), &.{HD}, 0.1, 1);
        try p.rand(try p.at(i, "attn.k_norm.weight"), &.{HD}, 0.1, 1);
        try p.rand(try p.at(i, "attn.wq_du.weight"), &.{ 2 * HD, H }, 0.1, 0);
        try p.rand(try p.at(i, "attn.wk_dv.weight"), &.{ HD, H }, 0.1, 0);
        try p.rand(try p.at(i, "attn.wv_dv.weight"), &.{ HD, H }, 0.1, 0);
        try p.rand(try p.at(i, "attn.wr_du.weight"), &.{ 2 * 4, H }, 0.1, 0);
        try p.rand(try p.at(i, "attn.wo_ud.weight"), &.{ H, 2 * HD }, 0.1, 0);
        try p.rand(try p.at(i, "attn.rel_logits_proj.proj"), &.{ 4, 8 }, 0.1, 0);
        try p.rand(try p.at(i, "attn.k_sconv.weight"), &.{ HD, K, 1 }, 0.1, 0);
        try p.rand(try p.at(i, "attn.v_sconv.weight"), &.{ HD, K, 1 }, 0.1, 0);
        try p.rand(try p.at(i, "attn_sconv.weight"), &.{ H, K, 1 }, 0.1, 0);
        try p.rand(try p.at(i, "mlp_sconv.weight"), &.{ H, K, 1 }, 0.1, 0);
        if (i == 0) {
            try p.rand(try p.at(i, "mlp.gate_proj.weight"), &.{ INTER, H }, 0.1, 0);
            try p.rand(try p.at(i, "mlp.up_proj.weight"), &.{ INTER, H }, 0.1, 0);
            try p.rand(try p.at(i, "mlp.down_proj.weight"), &.{ H, INTER }, 0.1, 0);
            try p.rand(try p.at(i, "mlp.global_scale"), &.{1}, 0.1, 1);
            continue;
        }
        try p.rand(try p.at(i, "mlp.gate.weight"), &.{ E + NS, H }, 0.1, 0);
        try p.rand(try p.at(i, "mlp.gate.bias"), &.{E}, 0.1, 0);
        try p.rand(try p.at(i, "mlp.gate.global_scale"), &.{1}, 0.1, 1);
        for ([_][]const u8{ "mlp.experts", "mlp.shared_experts" }, [_]u64{ E, NS }) |container, n| {
            try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.gate_proj.weight", .{container})), &.{ n, MI, H }, 0.1, 0);
            try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.up_proj.weight", .{container})), &.{ n, MI, H }, 0.1, 0);
            try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.down_proj.weight", .{container})), &.{ n, H, MI }, 0.1, 0);
        }
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"inkling_mm_model","eos_token_id":2,"text_config":{"model_max_length":4096,"hidden_size":64,
        \\"num_hidden_layers":2,"vocab_size":64,"unpadded_vocab_size":60,"num_attention_heads":2,"num_key_value_heads":1,
        \\"head_dim":32,"d_rel":4,"rel_extent":8,"log_scaling_n_floor":2,"log_scaling_alpha":0.1,"rms_norm_eps":1e-6,
        \\"local_layer_ids":[0],"dense_mlp_idx":1,"sconv_kernel_size":4,"logits_mup_width_multiplier":4.0,
        \\"sliding_window_size":4,"n_routed_experts":4,"num_experts_per_tok":2,"n_shared_experts":1,
        \\"dense_intermediate_size":128,"intermediate_size":32,"route_scale":2.0}}
        },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
