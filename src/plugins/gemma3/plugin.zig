//! Gemma 3 text decoders: `Decoder(TableEmbed, RmsNorm, PostNormed(Attention), PostNormed(GeGlu), ResidualTail,
//! SoftcapHead)` from `sdk/bases`, with this family's attention (sliding layers, the fused hd-256 decode step) and its
//! compiled activations. The load reads the engine's tensors through the pack's quant format.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
pub const config = @import("config.zig");
const Attention = load_mod.Attention;
pub const load_mod = @import("load.zig");
pub const compiled_mod = @import("compiled.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const extras = b.extras;

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = extras.PostNormed(Attention),
    .mlp = extras.PostNormed(extras.GeGlu),
    .seam = b.seam.ResidualTail,
    .head = extras.SoftcapHead,
});

comptime {
    if (Dense.batches_decode != config.batches_decode) @compileError("gemma3: config.batches_decode disagrees with the base");
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
    .weights = &.{},
    .spec = &.{ .trunk_weight = &load_mod.TrunkWeight(Loaded).get },
};

const gpa = load_mod.gpa;
const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;

/// Every term is the engine's default arithmetic over the cache layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A loaded model: the base's weights and every array handle it holds.
pub const Loaded = struct {
    base: bases.Model(Dense),
    layers: []Dense.Layer = &.{},
    owned: load_mod.Owned = .{},
    angles: qk_rope.AngleCache = .{},
    compiled: compiled_mod.Compiled = undefined,

    fn release(m: *Loaded) void {
        m.angles.deinit();
        m.compiled.deinit();
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
    var r = try load_mod.Reader.open(ctx, c);
    defer r.close();
    const s = r.s;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids } };
    m.compiled.init(s, c.final_logit_softcapping);
    errdefer m.release();
    m.layers = try gpa.alloc(Dense.Layer, c.num_hidden_layers);

    const o = &m.owned;
    const sh = try load_mod.AttnShared.build(o, c, s);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar)));
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        const li: u32 = @intCast(i);
        const attn = try load_mod.attentionAt(&r, o, c, &sh, li, scale, &m.angles);
        L.* = .{
            .input_norm = try r.norm(o, try r.name(&nb, "layers.{d}.input_layernorm.weight", .{li}), c.norm_has_offset),
            .mixer = .{ .inner = attn, .norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{li}), c.norm_has_offset) },
            .post_norm = try r.norm(o, try r.name(&nb, "layers.{d}.pre_feedforward_layernorm.weight", .{li}), c.norm_has_offset),
            .mlp = .{
                .inner = try load_mod.mlpAt(&r, o, c, li, attn.qkv.built(), &m.compiled),
                .norm = try r.norm(o, try r.name(&nb, "layers.{d}.post_feedforward_layernorm.weight", .{li}), c.norm_has_offset),
            },
        };
    }

    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try o.linear(embed);
    const emb_scale = load_mod.bf16Scalar(@sqrt(@as(f32, @floatFromInt(c.hidden_size))), s);
    try o.own(emb_scale);
    const final_norm = try r.norm(o, try r.name(&nb, "norm.weight", .{}), c.norm_has_offset);
    try mlx.check(mlx.mlx_array_eval(final_norm));
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = .bfloat16, .scale = emb_scale },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = .{
            .head = try load_mod.headOf(&r, o, embed, c.hidden_size, c.tie_word_embeddings),
            .cap = m.compiled.cap,
            .compiled = m.compiled.softcap,
        },
    };
    if (config.ternaryCandidate(c)) try load_mod.armTernary(.{ &m.layers, &m.base.weights.head.head }, null, load_mod.gatherOnly(&m.base.weights.head.head, embed), s);
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Dense, &m.base, f, out);
}

/// The tiny pack the conformance suite runs: four layers (sliding, global, sliding, global), width 64, seeded random
/// f32 weights, a tied head and a logit softcap.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const heads = 2;
    const kv_heads = 1;
    const hd = 32;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 3;
    const put = struct {
        fn f(al: std.mem.Allocator, list: *std.ArrayList(sdk.fixture.Tensor), sd: *u64, name: []const u8, shape: []const u64, scale: f32) !void {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            sd.* += 1;
            try list.append(al, .{ .name = name, .shape = shape, .data = try sdk.fixture.randomF32(al, sd.*, n, scale) });
        }
    }.f;
    try put(a, &tensors, &seed, "model.embed_tokens.weight", &.{ v, h }, 0.5);
    try put(a, &tensors, &seed, "model.norm.weight", &.{h}, 0.1);
    for (0..4) |i| {
        const L = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ li, leaf });
            }
        };
        for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight" }) |nm|
            try put(a, &tensors, &seed, try L.n(a, i, nm), &.{h}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.q_norm.weight"), &.{hd}, 0.1);
        try put(a, &tensors, &seed, try L.n(a, i, "self_attn.k_norm.weight"), &.{hd}, 0.1);
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
        \\{"model_type":"gemma3_text","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-6,
        \\ "rope_theta":1000000.0,"rope_local_base_freq":10000.0,"sliding_window":2,"sliding_window_pattern":2,
        \\ "query_pre_attn_scalar":32,"final_logit_softcapping":30.0,"max_position_embeddings":4096,"eos_token_id":1}
        },
        .{ .name = "generation_config.json", .bytes = "{\"top_k\":64,\"eos_token_id\":[1,106]}" },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
