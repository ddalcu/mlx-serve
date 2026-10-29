//! XHToken Spark-X2.5 (1.7B / 4B dense, hd 256): `Decoder(TableEmbed, RmsNorm, Attention, GatedGelu(.erf),
//! ResidualTail, LmHead)` with Spark's attention (`attention.zig`: 3 sliding + 1 full layers with per-type RoPE, a
//! per-head sigmoid output gate, the fused hd-256 RoPE at decode). The load splits the checkpoint's one
//! `q_k_v_proj` into q, k and v as the engine's loader did, then joins q|k|v|g rows of a quantized layer.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
const loader = @import("../llama/load.zig");
const Attention = @import("attention.zig").Attention;
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;

pub const Dense = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = Attention,
    .mlp = b.extras.GatedGelu(.erf),
    .seam = b.seam.ResidualTail,
    .head = b.head.LmHead,
});

comptime {
    if (Dense.batches_decode != config.batches_decode) @compileError("spark2_5: config.batches_decode disagrees with the base");
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

pub const Loaded = struct {
    base: bases.Model(Dense),
    layers: []Dense.Layer = &.{},
    owned: loader.Owned = .{},
    angles: qk_rope.AngleCache = .{},

    fn release(m: *Loaded) void {
        m.angles.deinit();
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

/// Rows `[from, to)` of `a` (rank 2) as their own array: transposed to `[in, out]` for a dense weight (`transpose`),
/// else contiguous.
fn rowsOf(a: Array, from: c_int, to: c_int, transpose: bool, s: mlx.mlx_stream) !Array {
    if (a.ctx == null) return a;
    const cols = mlx.getShape(a)[1];
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_slice(&view, a, &[_]c_int{ from, 0 }, 2, &[_]c_int{ to, cols }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var part = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(part);
    if (transpose)
        try mlx.check(mlx.mlx_transpose_axes(&part, view, &[_]c_int{ 1, 0 }, 2, s))
    else
        try mlx.check(mlx.mlx_contiguous(&part, view, false, s));
    return part;
}

/// The checkpoint's one `q_k_v_proj` split into q, k and v (row slices, each its own array), as the engine split it;
/// the caller owns them.
fn splitQkv(r: *loader.Reader, base: []const u8, c: *const config.Config) ![3]Linear {
    const q_rows: c_int = @intCast(c.num_attention_heads * c.head_dim);
    const kv_rows: c_int = @intCast(c.num_key_value_heads * c.head_dim);
    const bounds = [_]c_int{ 0, q_rows, q_rows + kv_rows, q_rows + 2 * kv_rows };
    var nb: [256]u8 = undefined;
    const raw = r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
    if (raw.ctx == null) return error.MissingWeight;
    const w: Array = .{ .ctx = raw.ctx };
    if (mlx.getShape(w).len != 2 or mlx.getShape(w)[0] != bounds[3]) return error.BadFusedQkvShape;
    var out: [3]Linear = @splat(.{});
    errdefer for (out) |l| freeLinear(l);
    if (r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.scales", .{base})).ctx == null) {
        for (&out, 0..) |*l, i| l.* = .{ .w = try rowsOf(w, bounds[i], bounds[i + 1], true, r.s) };
        return out;
    }
    // The quant params of the whole projection hold for every row slice of it.
    const full = try r.proj(base, c.hidden_size);
    defer for ([_]Array{ full.w, full.s, full.b }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for (&out, 0..) |*l, i| {
        l.* = .{ .bits = full.bits, .group_size = full.group_size, .mode = full.mode };
        l.w = try rowsOf(full.w, bounds[i], bounds[i + 1], false, r.s);
        l.s = try rowsOf(full.s, bounds[i], bounds[i + 1], false, r.s);
        l.b = try rowsOf(full.b, bounds[i], bounds[i + 1], false, r.s);
    }
    return out;
}

fn freeLinear(l: Linear) void {
    for ([_]Array{ l.w, l.s, l.b }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{"model"}, "embedding.weight");
    defer r.close();
    const s = r.s;

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s } };
    errdefer m.release();
    m.layers = try gpa.alloc(Dense.Layer, c.num_hidden_layers);
    const o = &m.owned;

    const hidden = c.hidden_size;
    const q_rows = c.num_attention_heads * c.head_dim;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        var qkv = try splitQkv(&r, try r.name(&nb, "layers.{d}.self_attn.q_k_v_proj", .{i}), c);
        var gw = r.proj(try r.name(&nb, "layers.{d}.self_attn.g_proj", .{i}), hidden) catch |e| {
            for (qkv) |l| freeLinear(l);
            return e;
        };
        // A quantized layer joins q|k|v|g (the join frees the split copies, leaving row views of its buffer).
        const joined = if (qkv[0].s.ctx != null) try loader.join(o, &.{ &qkv[0], &qkv[1], &qkv[2], &gw }, s) else b.linear.Joined{};
        for ([_]Linear{ qkv[0], qkv[1], qkv[2], gw }) |l| try o.linear(l);
        const ow = try r.proj(try r.name(&nb, "layers.{d}.self_attn.out_proj", .{i}), q_rows);
        try o.linear(ow);
        var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
        var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
        const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
        const gu = if (gate.s.ctx != null) try loader.join(o, &.{ &gate, &up }, s) else b.linear.Joined{};
        for ([_]Linear{ gate, up, down }) |l| try o.linear(l);
        const input_norm = try r.need(try r.name(&nb, "layers.{d}.input_layernorm.weight", .{i}));
        try o.own(input_norm);
        const post_norm = try r.need(try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{i}));
        try o.own(post_norm);
        const global = c.global[i];
        L.* = .{
            .input_norm = input_norm,
            .post_norm = post_norm,
            .mixer = .{
                .q = qkv[0],
                .k = qkv[1],
                .v = qkv[2],
                .o = ow,
                .qkv = joined,
                .out_gate = gw,
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = c.scale,
                    .rope_dims = @intCast(c.ropeDims(global)),
                    .rope_base = if (global) c.rope_theta else c.rope_local_base_freq,
                    .rope_scale = 1.0,
                },
                .window = if (global) 0 else c.sliding_window,
                .angles = &m.angles,
                .family = @intFromBool(global),
            },
            .mlp = .{ .gate = gate, .up = up, .down = down, .gu = gu, .rows_joined = joined.built() },
        };
    }

    const embed = try r.tableAt(try r.name(&nb, "embedding", .{}), hidden);
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
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Dense, &m.base, f, out);
}

/// The tiny pack the conformance suite runs: four layers (three sliding, one full), width 64, seeded random f32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const v = 64;
    const h = 64;
    const inter = 128;
    const heads = 2;
    const kv_heads = 1;
    const hd = 32;
    var tensors: std.ArrayList(sdk.fixture.Tensor) = .empty;
    var seed: u64 = 7;
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
    try put(a, &tensors, &seed, "model.embedding.weight", &.{ v, h }, 0.5);
    try tensors.append(a, .{ .name = "model.norm.weight", .shape = &.{h}, .data = ones });
    for (0..4) |i| {
        const n = struct {
            fn n(al: std.mem.Allocator, li: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ li, leaf });
            }
        }.n;
        try tensors.append(a, .{ .name = try n(a, i, "input_layernorm.weight"), .shape = &.{h}, .data = ones });
        try tensors.append(a, .{ .name = try n(a, i, "post_attention_layernorm.weight"), .shape = &.{h}, .data = ones });
        try put(a, &tensors, &seed, try n(a, i, "self_attn.q_k_v_proj.weight"), &.{ (heads + 2 * kv_heads) * hd, h }, 0.1);
        try put(a, &tensors, &seed, try n(a, i, "self_attn.out_proj.weight"), &.{ h, heads * hd }, 0.1);
        try put(a, &tensors, &seed, try n(a, i, "self_attn.g_proj.weight"), &.{ heads, h }, 0.1);
        try put(a, &tensors, &seed, try n(a, i, "mlp.gate_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try n(a, i, "mlp.up_proj.weight"), &.{ inter, h }, 0.1);
        try put(a, &tensors, &seed, try n(a, i, "mlp.down_proj.weight"), &.{ h, inter }, 0.1);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"spark2_5","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":2,
        \\ "num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-6,"sliding_window":4,
        \\ "layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
        \\ "rope_parameters":{"full_attention":{"rope_theta":5000000.0,"partial_rotary_factor":0.25},
        \\ "sliding_attention":{"rope_theta":10000.0}},"tie_word_embeddings":true,"max_position_embeddings":4096,
        \\ "eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
