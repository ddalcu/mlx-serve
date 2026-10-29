//! BERT encoders (`bert`, the mlx-community embedding conversions): the encoder base over the engine's loaded
//! tensors. Word + position + token-type (0) rows under a LayerNorm, then post-norm layers of bidirectional attention
//! and a tanh-GELU MLP (the compiled closure the engine compiles at load), all projections with additive biases. The
//! forward returns the hidden rows; the engine pools them (`EmbedFacts.pooling`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const reader = @import("../../sdk/bases/reader.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Ctx = bases.Ctx;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

/// Word + position + token-type rows (type 0), LayerNorm.
const BertEmbed = struct {
    pub const kind: bases.block.Kind = .embed;
    pub const supports_batched_rows = false;
    pub const Weights = struct { word: Linear, pos: Linear, toktype: Linear, norm: b.norm.LayerNorm.Weights, hidden: c_int };

    pub fn forward(w: *const Weights, c: *Ctx, ids: Array) !Array {
        const g = c.g;
        const sh = ops.shape(ids);
        const flat = try g.reshape(ids, &.{sh[0] * sh[1]});
        const word = try g.reshape(try b.embed.gatherRows(g, &w.word, flat, .bfloat16), &.{ sh[0], sh[1], w.hidden });
        const row = [_]c_int{ 1, sh[1], w.hidden };
        const pos = try g.reshape(try b.embed.gatherRows(g, &w.pos, try g.arange(0, @floatFromInt(sh[1]), 1, .int32), .bfloat16), &row);
        const tt = try g.reshape(try b.embed.gatherRows(g, &w.toktype, try g.zeros(&.{sh[1]}, .int32), .bfloat16), &row);
        return b.norm.LayerNorm.forward(&w.norm, c, try g.add(try g.add(word, pos), tt));
    }
};

/// `out(gelu_tanh(inter(x)))`, the GELU through the model's compiled closure (compiling is not output-preserving).
const GeluMlp = struct {
    pub const kind: bases.block.Kind = .mlp;
    pub const supports_batched_rows = false;
    pub const Weights = struct { inter: Linear, out: Linear, gelu: compiled.Closure = .{ .ctx = null } };

    pub fn forward(w: *const Weights, c: *Ctx, x: Array) !Array {
        const g = c.g;
        const h = try w.inter.forward(g, x);
        const act = if (w.gelu.ctx == null) try b.mlp.geluTanh(g, h) else ((try compiled.apply(g, w.gelu, &.{h}, 1)) orelse return error.ClosureOutputs)[0];
        return w.out.forward(g, act);
    }
};

const GeluPayload = struct { s: ops.Stream };

fn geluBody(p: *const GeluPayload, in: []const Array, out: *[1]Array) anyerror!void {
    var g = ops.Graph.init(gpa, p.s);
    defer g.deinit();
    out[0] = g.escape(try b.mlp.geluTanh(&g, in[0]));
}

pub const Model = bases.Encoder(.{
    .embed = BertEmbed,
    .attention = b.attention.BidirAttention,
    .mlp = GeluMlp,
    .norm = b.norm.LayerNorm,
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
    .weights = &.{},
};

/// No KV: an encoder's activations are its whole bill, the engine's default arithmetic.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},
    gelu: compiled.Closure = .{ .ctx = null },

    fn release(m: *Loaded) void {
        compiled.free(&m.gelu);
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
    var r = try reader.Reader.open(ctx, c.block_json, "");
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.layer_norm_eps, c.layer_norm_eps), .a = gpa, .s = r.s } };
    errdefer m.release();
    m.gelu = compiled.compile(GeluPayload, .{ .s = r.s }, 1, geluBody, true);
    const hidden = c.hidden_size;
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| {
        var nb: [128]u8 = undefined;
        const at = struct {
            fn f(buf: []u8, layer: usize, leaf: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "encoder.layer.{d}.{s}", .{ layer, leaf });
            }
        }.f;
        L.* = .{
            .attention = .{
                .q = try lin(m, &r, try at(&nb, i, "attention.self.query"), hidden),
                .k = try lin(m, &r, try at(&nb, i, "attention.self.key"), hidden),
                .v = try lin(m, &r, try at(&nb, i, "attention.self.value"), hidden),
                .o = try lin(m, &r, try at(&nb, i, "attention.output.dense"), hidden),
                .heads = @intCast(c.num_attention_heads),
                .head_dim = @intCast(c.head_dim),
            },
            .attn_norm = try norm(m, &r, try at(&nb, i, "attention.output.LayerNorm")),
            .mlp = .{
                .inter = try lin(m, &r, try at(&nb, i, "intermediate.dense"), hidden),
                .out = try lin(m, &r, try at(&nb, i, "output.dense"), c.intermediate_size),
                .gelu = m.gelu,
            },
            .mlp_norm = try norm(m, &r, try at(&nb, i, "output.LayerNorm")),
        };
    }
    m.base.weights = .{
        .embed = .{
            .word = try tbl(m, &r, "embeddings.word_embeddings", hidden),
            .pos = try tbl(m, &r, "embeddings.position_embeddings", hidden),
            .toktype = try tbl(m, &r, "embeddings.token_type_embeddings", hidden),
            .norm = try norm(m, &r, "embeddings.LayerNorm"),
            .hidden = @intCast(hidden),
        },
        .layers = m.layers,
    };
    return m;
}

/// A projection with its additive `.bias`.
fn lin(m: *Loaded, r: *reader.Reader, base: []const u8, in_dim: u32) !Linear {
    var l = try r.proj(base, in_dim);
    try m.owned.linear(l);
    var nb: [160]u8 = undefined;
    l.bias = try r.need(try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    try m.owned.own(l.bias);
    return l;
}

fn tbl(m: *Loaded, r: *reader.Reader, base: []const u8, in_dim: u32) !Linear {
    const l = try r.table(base, in_dim);
    try m.owned.linear(l);
    return l;
}

fn norm(m: *Loaded, r: *reader.Reader, base: []const u8) !b.norm.LayerNorm.Weights {
    var nb: [160]u8 = undefined;
    const w = try r.need(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
    try m.owned.own(w);
    const bias = try r.need(try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    try m.owned.own(bias);
    return .{ .w = w, .b = bias };
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    if (!abi.declaresMode(.{ .embed = true }, f.mode)) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

/// A 4-bit group-64 projection `<name>.{weight,scales,biases}` of `[rows, cols]`: random codes, small scales.
fn quantized(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, rows: u64, cols: u64, scale: f32) !void {
    try t.words(fmt ++ ".weight", args, &.{ rows, cols / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ rows, cols / 64 }, .BF16, scale / 8);
    try t.fill(fmt ++ ".biases", args, &.{ rows, cols / 64 }, .BF16, -scale);
}

/// The tiny pack the conformance suite runs: one layer, width 64, 4-bit (the engine's BERT load is quantized-only).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    var t = sdk.tensors.Tensors.init(a, 1);
    try quantized(&t, "embeddings.word_embeddings", .{}, 64, h, 0.5);
    try quantized(&t, "embeddings.position_embeddings", .{}, 32, h, 0.5);
    try quantized(&t, "embeddings.token_type_embeddings", .{}, 2, h, 0.5);
    try t.fill("embeddings.LayerNorm.weight", .{}, &.{h}, .F32, 1);
    try t.fill("embeddings.LayerNorm.bias", .{}, &.{h}, .F32, 0);
    for ([_][]const u8{ "attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense" }) |p| {
        try quantized(&t, "encoder.layer.0.{s}", .{p}, h, h, 0.1);
        try t.rand("encoder.layer.0.{s}.bias", .{p}, &.{h}, .F32, 0.1);
    }
    try quantized(&t, "encoder.layer.0.intermediate.dense", .{}, 128, h, 0.1);
    try t.rand("encoder.layer.0.intermediate.dense.bias", .{}, &.{128}, .F32, 0.1);
    try quantized(&t, "encoder.layer.0.output.dense", .{}, h, 128, 0.1);
    try t.rand("encoder.layer.0.output.dense.bias", .{}, &.{h}, .F32, 0.1);
    for ([_][]const u8{ "attention.output.LayerNorm", "output.LayerNorm" }) |p| {
        try t.fill("encoder.layer.0.{s}.weight", .{p}, &.{h}, .F32, 1);
        try t.fill("encoder.layer.0.{s}.bias", .{p}, &.{h}, .F32, 0);
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"bert","vocab_size":64,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,
        \\ "num_attention_heads":2,"max_position_embeddings":32,"type_vocab_size":2,"layer_norm_eps":1e-12,
        \\ "quantization":{"group_size":64,"bits":4}}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
}
