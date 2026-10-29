//! OpenAI gpt-oss (20B-A3.6B / 120B-A5.1B): `Decoder(TableEmbed, RmsNorm, DecodeCopies(SinkAttention), Moe, AddNorm,
//! LmHead)`:
//! sink attention with additive biases and YaRN on every layer, alternating sliding(128)/full layers, and routed
//! experts with additive router and per-expert biases under the clamped SwiGLU. Serial, as the legacy.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const computeYarnFreqs = @import("../../sdk/blocks/rope/yarn.zig").computeYarnFreqs;
const DecodeCopies = @import("../../sdk/bases/blocks/decode_copies.zig").DecodeCopies;
const loader = @import("../llama/load.zig");
const SinkAttention = @import("attention.zig").SinkAttention;
const moe = @import("moe.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Model = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = DecodeCopies(SinkAttention),
    .mlp = moe.Moe,
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
    .cache = &.{},
    .weights = &.{},
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded) },
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
    route_fns: []compiled.Closure = &.{},

    fn release(m: *Loaded) void {
        for (m.route_fns) |*f| compiled.free(f);
        gpa.free(m.route_fns);
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

/// The YaRN denominators and the f32 `[head_dim]` attention-factor table, as the engine's init builds them (whole
/// head rotated; the engine reads the YaRN block's `truncate` only from `rope_parameters`, so it is always on here).
fn yarnTables(o: *loader.Owned, c: *const config.Config) ![2]Array {
    const y = c.yarn.?;
    const half: usize = c.head_dim / 2;
    const f64s = try gpa.alloc(f64, half);
    defer gpa.free(f64s);
    computeYarnFreqs(f64s, c.head_dim, 1.0, c.rope_theta, y.factor, y.beta_fast, y.beta_slow, y.orig_max, true);
    const f32s = try gpa.alloc(f32, c.head_dim);
    defer gpa.free(f32s);
    for (f64s, 0..) |v, i| f32s[i] = @floatCast(v);
    const freqs = mlx.mlx_array_new_data(f32s.ptr, &[_]c_int{@intCast(half)}, 1, .float32);
    try o.own(freqs);
    @memset(f32s, y.attention_factor);
    const mscale = mlx.mlx_array_new_data(f32s.ptr, &[_]c_int{@intCast(c.head_dim)}, 1, .float32);
    try o.own(mscale);
    return .{ freqs, mscale };
}

/// A projection with its additive `.bias` (beside the quantizer's `.biases`), owned by `o`.
fn biased(r: *loader.Reader, o: *loader.Owned, nb: []u8, i: usize, comptime leaf: []const u8, in_dim: u32) !Linear {
    var l = try r.proj(try r.name(nb, "layers.{d}." ++ leaf, .{i}), in_dim);
    l.bias = try r.share(try r.name(nb, "layers.{d}." ++ leaf ++ ".bias", .{i}));
    try o.linear(l);
    return l;
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{"model"}, "embed_tokens.weight");
    defer r.close();
    const s = r.s;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    m.route_fns = try gpa.alloc(compiled.Closure, c.num_hidden_layers);
    @memset(m.route_fns, .{ .ctx = null });
    const o = &m.owned;
    const yarn: ?[2]Array = if (c.yarn != null) try yarnTables(o, c) else null;

    const hidden = c.hidden_size;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        const global = c.global[i];
        var sinks = try r.share(try r.name(&nb, "layers.{d}.self_attn.sinks", .{i}));
        if (sinks.ctx == null) sinks = try r.share(try r.name(&nb, "layers.{d}.self_attn.attention_sink_bias", .{i}));
        try o.own(sinks);
        const inner: SinkAttention.Weights = .{
            .q = try biased(&r, o, &nb, i, "self_attn.q_proj", hidden),
            .k = try biased(&r, o, &nb, i, "self_attn.k_proj", hidden),
            .v = try biased(&r, o, &nb, i, "self_attn.v_proj", hidden),
            .o = try biased(&r, o, &nb, i, "self_attn.o_proj", c.num_attention_heads * c.head_dim),
            .sinks = sinks,
            .geo = .{
                .heads = @intCast(c.num_attention_heads),
                .kv_heads = @intCast(c.num_key_value_heads),
                .head_dim = @intCast(c.head_dim),
                .scale = c.scale,
                .rope_dims = @intCast(c.head_dim),
                .rope_base = c.rope_theta,
                .rope_scale = 1.0,
                .freqs = if (yarn) |y| y[0] else .{},
            },
            .mscale = if (yarn) |y| y[1] else .{},
            .window = if (global) 0 else c.sliding_window,
        };
        var dqs: [4]Linear = undefined;
        for (&dqs, [_]Linear{ inner.q, inner.k, inner.v, inner.o }) |*d, l| {
            d.* = try b.decode_copies.decodeCopy(l, @intCast(i), c.num_hidden_layers, s);
            try o.linear(d.*);
        }
        var bank: [3]b.moe.Bank = undefined;
        var bias: [3]Array = undefined;
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |nm, j| {
            bank[j] = try r.bank(try r.name(&nb, "layers.{d}.mlp.experts." ++ nm, .{i}), if (j == 2) c.intermediate_size else hidden);
            for ([_]Array{ bank[j].w, bank[j].s, bank[j].b }) |a| try o.own(a);
            // The fused decode arms have no bias slot; this block runs only the biased sorted path.
            bias[j] = try r.need(try r.name(&nb, "layers.{d}.mlp.experts." ++ nm ++ ".bias", .{i}));
            try o.own(bias[j]);
        }
        var w: moe.Moe.Weights = .{
            .router = try biased(&r, o, &nb, i, "mlp.router", hidden),
            .gate = bank[0],
            .up = bank[1],
            .down = bank[2],
            .gate_bias = bias[0],
            .up_bias = bias[1],
            .down_bias = bias[2],
            .top_k = @intCast(c.top_k),
            .experts = @intCast(c.num_experts),
            .limit = c.swiglu_limit,
        };
        m.route_fns[i] = moe.compileRouting(&w, s);
        w.route_fn = m.route_fns[i];
        L.* = .{
            .input_norm = try r.need(try r.name(&nb, "layers.{d}.input_layernorm.weight", .{i})),
            .mixer = .{ .inner = inner, .dq = dqs },
            .post_norm = try r.need(try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{i})),
            .mlp = w,
        };
        try o.own(L.input_norm);
        try o.own(L.post_norm);
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), hidden);
    try o.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try o.own(final_norm);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = try loader.headOf(&r, o, embed, hidden, c.tie_word_embeddings),
    };
    return m;
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

/// The tiny pack the conformance suite runs: two layers (a sliding one, a full one), width 64, 4 experts of 64
/// (top 2), YaRN, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const fx = sdk.fixture;
    const V = 64;
    const H = 64;
    const HD = 32;
    const HEADS = 2;
    const E = 4;
    const I = 64;
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 31;
    const ones = try a.alloc(f32, H);
    @memset(ones, 1);
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
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{H}, .data = ones });
    for (0..2) |i| {
        try tensors.append(a, .{ .name = try p.at(i, "input_layernorm.weight"), .shape = &.{H}, .data = ones });
        try tensors.append(a, .{ .name = try p.at(i, "post_attention_layernorm.weight"), .shape = &.{H}, .data = ones });
        inline for (.{ .{ "q_proj", HEADS * HD, H }, .{ "k_proj", HD, H }, .{ "v_proj", HD, H }, .{ "o_proj", H, HEADS * HD } }) |t| {
            try p.rand(try p.at(i, "self_attn." ++ t[0] ++ ".weight"), &.{ t[1], t[2] }, 0.1);
            try p.rand(try p.at(i, "self_attn." ++ t[0] ++ ".bias"), &.{t[1]}, 0.1);
        }
        try p.rand(try p.at(i, "self_attn.sinks"), &.{HEADS}, 1.0);
        try p.rand(try p.at(i, "mlp.router.weight"), &.{ E, H }, 0.1);
        try p.rand(try p.at(i, "mlp.router.bias"), &.{E}, 0.1);
        inline for (.{ .{ "gate_proj", I, H }, .{ "up_proj", I, H }, .{ "down_proj", H, I } }) |t| {
            try p.rand(try p.at(i, "mlp.experts." ++ t[0] ++ ".weight"), &.{ E, t[1], t[2] }, 0.1);
            try p.rand(try p.at(i, "mlp.experts." ++ t[0] ++ ".bias"), &.{ E, t[1] }, 0.1);
        }
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"gpt_oss","hidden_size":64,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":2,
        \\"num_key_value_heads":1,"head_dim":32,"num_local_experts":4,"experts_per_token":2,"rms_norm_eps":1e-05,
        \\"layer_types":["sliding_attention","full_attention"],"max_position_embeddings":4096,
        \\"rope_scaling":{"beta_fast":32.0,"beta_slow":1.0,"factor":32.0,"original_max_position_embeddings":128,
        \\"rope_type":"yarn"},"rope_theta":150000,"sliding_window":4,"swiglu_limit":7.0,"tie_word_embeddings":false,
        \\"vocab_size":64,"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
