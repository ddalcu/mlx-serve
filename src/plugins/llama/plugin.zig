//! Llama, Mistral and Qwen2: the dense decoder base over the engine's loaded tensors. The forward is
//! `Decoder(TableEmbed, RmsNorm, Attention, SwiGlu, ResidualTail, LmHead)`; the load reads each projection through the
//! pack's quant format, joins q|k|v and gate|up rows of quantized layers (as the engine's loader did) and builds the
//! scaled rope (`sdk/config/rope_scaling.zig`). Every layer is full attention at the shell's geometry (no cache layout hook).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const scaled = @import("../../sdk/blocks/rope/scaled.zig");
pub const loader = @import("load.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.attention.Attention,
    .mlp = b.mlp.SwiGlu,
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

comptime {
    if (Dense.batches_decode != config.batches_decode) @compileError("llama: config.batches_decode disagrees with the base");
}

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &Table(Dense).unload,
    .forward = &Table(Dense).forward,
    .cache = &.{},
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded(Dense)) },
    .weights = &.{},
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;

/// Every term is the engine's default arithmetic over the shell (full attention at the shell's geometry).
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A loaded model of decoder `D`: the base's weights and every array handle it holds.
pub fn Loaded(comptime D: type) type {
    return struct {
        base: bases.Model(D),
        layers: []D.Layer = &.{},
        owned: loader.Owned = .{},

        pub fn release(m: *@This()) void {
            m.owned.release();
            gpa.free(m.layers);
            m.base.shared.deinit();
        }
    };
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadDense(Dense, ctx, plainNorm) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn plainNorm(_: *loader.Owned, w: Array, _: *const config.Config, _: mlx.mlx_stream) !Array {
    return w;
}

/// The weight-name prefixes mlx-lm writes for a dense decoder.
pub const prefixes = [_][]const u8{ "model", "language_model.model", "model.language_model" };

/// Loads a dense decoder `D` over this family's layout (`Config`); `normOf` wraps each RMS norm weight (owned by
/// the model already) into `D`'s norm weights.
pub fn loadDense(comptime D: type, ctx: *const abi.LoadCtx, comptime normOf: anytype) !*Loaded(D) {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &prefixes, "embed_tokens.weight");
    defer r.close();
    const s = r.s;

    const M = Loaded(D);
    const m = try gpa.create(M);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s } };
    errdefer m.release();
    m.layers = try gpa.alloc(D.Layer, c.num_hidden_layers);
    const o = &m.owned;

    scaled.logScaling(ctx.parsed.shell.model_type.slice(), c.rope);
    const freqs: Array = (try scaled.freqs(c.rope, c.head_dim, c.rope_theta)) orelse .{};
    if (freqs.ctx != null) try o.own(freqs);

    const hidden = c.hidden_size;
    const q_rows = c.num_attention_heads * c.head_dim;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
        var k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
        var v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
        // Qwen2's additive q/k/v biases.
        q.bias = try r.share(try r.name(&nb, "layers.{d}.self_attn.q_proj.bias", .{i}));
        k.bias = try r.share(try r.name(&nb, "layers.{d}.self_attn.k_proj.bias", .{i}));
        v.bias = try r.share(try r.name(&nb, "layers.{d}.self_attn.v_proj.bias", .{i}));
        const ow = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), q_rows);
        var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
        var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
        const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
        const qkv = if (q.s.ctx != null and q.bias.ctx == null and k.bias.ctx == null and v.bias.ctx == null)
            try loader.join(o, &.{ &q, &k, &v }, s)
        else
            b.linear.Joined{};
        const gu = if (gate.s.ctx != null) try loader.join(o, &.{ &gate, &up }, s) else b.linear.Joined{};
        for ([_]Linear{ q, k, v, ow, gate, up, down }) |l| try o.linear(l);
        const input_norm = try r.need(try r.name(&nb, "layers.{d}.input_layernorm.weight", .{i}));
        try o.own(input_norm);
        const post_norm = try r.need(try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{i}));
        try o.own(post_norm);
        L.* = .{
            .input_norm = try normOf(o, input_norm, c, s),
            .post_norm = try normOf(o, post_norm, c, s),
            .mixer = .{
                .q = q,
                .k = k,
                .v = v,
                .o = ow,
                .qkv = qkv,
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
                    .rope_dims = @intCast(c.head_dim),
                    .rope_base = c.rope_theta,
                    .rope_scale = rope_scaling.positionScale(c.rope),
                    .freqs = freqs,
                    .mscale = @floatCast(rope_scaling.mscale(c.rope)),
                    .ntk = rope_scaling.ntk(c.rope, c.max_position_embeddings),
                },
            },
            .mlp = .{ .gate = gate, .up = up, .down = down, .gu = gu, .rows_joined = qkv.built() },
        };
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), hidden);
    try o.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try o.own(final_norm);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = try normOf(o, final_norm, c, s),
        .head = try loader.headOf(&r, o, embed, hidden, c.tie_word_embeddings),
    };
    return m;
}

/// `ModelTable.unload` and `forward` over a model of decoder `D`.
pub fn Table(comptime D: type) type {
    return struct {
        pub fn unload(mp: *anyopaque) callconv(.c) void {
            const m: *Loaded(D) = @ptrCast(@alignCast(mp));
            m.release();
            gpa.destroy(m);
        }

        pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
            const m: *Loaded(D) = @ptrCast(@alignCast(mp));
            return run_mod.tableForward(D, &m.base, f, out);
        }

    };
}

/// The tiny pack the conformance suite runs: two layers, width 64, seeded random f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const heads = 2;
    const kv_heads = 1;
    const hd = 32;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 1;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = shape, .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    const ones = try a.alloc(f32, h);
    @memset(ones, 1);
    try put(a, &tensors, &seed, "model.embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, "lm_head.weight", &.{ v, h }, 0.1);
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{h}, .data = ones });
    for (0..2) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        try tensors.append(a, .{ .name = try L.n(a, i, "input_layernorm.weight"), .shape = &.{h}, .data = ones });
        try tensors.append(a, .{ .name = try L.n(a, i, "post_attention_layernorm.weight"), .shape = &.{h}, .data = ones });
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_proj.weight"), &.{ heads * hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.v_proj.weight"), &.{ kv_heads * hd, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.o_proj.weight"), &.{ h, heads * hd }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "mlp.down_proj.weight"), &.{ h, inter }, 0.1);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"llama","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
        \\ "num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-5,"rope_theta":10000.0,
        \\ "max_position_embeddings":4096,"hidden_act":"silu","eos_token_id":2}
        },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.6,\"eos_token_id\":[2,3]}" },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
