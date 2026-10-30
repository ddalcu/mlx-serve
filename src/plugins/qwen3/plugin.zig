//! Qwen3 dense and MoE trunks over the engine's loaded tensors. Dense (`qwen3`) is the decoder base with per-head q/k
//! norms: `Decoder(TableEmbed, RmsNorm, Attention, SwiGlu, ResidualTail, LmHead)`, the engine's dense forward. The MoE
//! layout (`qwen3_moe`) rides the engine's layered forward: `Decoder(TableEmbed, RmsNorm, GatedAttention (no output
//! gate), Moe (softmax top-k, no shared expert), AddNorm, LmHead)`; it decodes serial, as the engine serves it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const moe_affine4 = @import("../../sdk/blocks/moe/moe_affine4.zig");
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const ld = @import("load.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const scaled = @import("../../sdk/blocks/rope/scaled.zig");
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

pub const Moe = bases.Decoder(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .mixer = b.gated_attention.GatedAttention,
    .mlp = b.moe.Moe,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

comptime {
    if (!Dense.batches_decode) @compileError("qwen3: the dense base no longer batches");
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
    .spec = &.{ .trunk_weight = &trunkWeight },
    .weights = &.{},
};

const gpa = ld.gpa;
const Verdict = abi.Verdict;
const Array = mlx.mlx_array;
const Linear = ld.Linear;

/// The embedding table and head the drafters read, borrowed (a tied head is the table).
fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return switch (m.body) {
        inline .dense, .moe => |*d| ld.trunkWeight(switch (which) {
            .embed => &d.base.weights.embed.table,
            .head => &d.base.weights.head,
        }, out),
    };
}

/// Every term is the engine's default arithmetic over the shell (full attention at the shell's geometry).
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

/// A loaded model: one of the two bases, and every handle it holds.
pub const Loaded = struct {
    caps: abi.Caps,
    body: union(enum) {
        dense: struct { base: bases.Model(Dense), layers: []Dense.Layer },
        moe: struct { base: bases.Model(Moe), layers: []Moe.Layer, routes: []compiled.Closure, angles: b.qk_rope.AngleCache = .{} },
    },
    owned: ld.Owned = .{},

    fn release(m: *Loaded) void {
        switch (m.body) {
            .dense => |*d| {
                gpa.free(d.layers);
                d.base.shared.deinit();
            },
            .moe => |*e| {
                for (e.routes) |*c| compiled.free(c);
                gpa.free(e.routes);
                e.angles.deinit();
                gpa.free(e.layers);
                e.base.shared.deinit();
            },
        }
        m.owned.deinit();
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = ld.failed(e);
        return null;
    };
}

/// The trunk's scaled spectrum (null = the base's), owned by `m`.
fn ropeFreqs(m: *Loaded, c: *const config.Config) !Array {
    const f = (try scaled.freqs(c.rope, c.head_dim, c.rope_theta)) orelse return .{};
    try m.owned.own(f);
    return f;
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const c = config.of(ctx.parsed);
    const src = ctx.weights orelse return error.NoWeightSource;
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    var r: ld.Reader = undefined;
    try r.open(src, c.block_json, ld.probePrefix(src, "model") orelse return error.MissingWeight, s);
    defer r.close();

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .caps = ctx.parsed.caps, .body = undefined };
    errdefer m.owned.deinit();
    const shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps);
    scaled.logScaling(plugin.name, c.rope);
    if (c.moe) {
        m.body = .{ .moe = .{ .base = .{ .weights = undefined, .shared = shared, .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids }, .layers = &.{}, .routes = &.{} } };
        errdefer m.release();
        try loadMoe(m, &r, c, s);
    } else {
        m.body = .{ .dense = .{ .base = .{ .weights = undefined, .shared = shared, .a = gpa, .s = s, .placeholders = ctx.parsed.placeholder_ids }, .layers = &.{} } };
        errdefer m.release();
        try loadDense(m, &r, c, s);
    }
    return m;
}

fn norm(m: *Loaded, r: *ld.Reader, comptime leaf: []const u8, i: usize) !Array {
    var nb: [256]u8 = undefined;
    const a = try r.need(try r.name(&nb, "layers.{d}." ++ leaf, .{i}));
    try m.owned.own(a);
    return a;
}

fn loadDense(m: *Loaded, r: *ld.Reader, c: *const config.Config, s: mlx.mlx_stream) !void {
    const d = &m.body.dense;
    d.layers = try gpa.alloc(Dense.Layer, c.num_hidden_layers);
    const freqs = try ropeFreqs(m, c);
    const hidden = c.hidden_size;
    const q_rows = c.num_attention_heads * c.head_dim;
    var nb: [256]u8 = undefined;
    for (d.layers, 0..) |*L, i| {
        var q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
        var k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
        var v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
        const o = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), q_rows);
        var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
        var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
        const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
        const qkv = if (q.s.ctx != null) try ld.join(&m.owned, &.{ &q, &k, &v }, s) else b.linear.Joined{};
        const gu = if (gate.s.ctx != null) try ld.join(&m.owned, &.{ &gate, &up }, s) else b.linear.Joined{};
        for ([_]Linear{ q, k, v, o, gate, up, down }) |l| try m.owned.linear(l);
        L.* = .{
            .input_norm = try norm(m, r, "input_layernorm.weight", i),
            .post_norm = try norm(m, r, "post_attention_layernorm.weight", i),
            .mixer = .{
                .q = q,
                .k = k,
                .v = v,
                .o = o,
                .q_norm = try norm(m, r, "self_attn.q_norm.weight", i),
                .k_norm = try norm(m, r, "self_attn.k_norm.weight", i),
                .qkv = qkv,
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar))),
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
    d.base.weights = try trunk(Dense, m, r, c, d.layers);
}

/// Embedding, final norm and head, as the engine binds them.
fn trunk(comptime D: type, m: *Loaded, r: *ld.Reader, c: *const config.Config, layers: []const D.Layer) !D.Weights {
    var nb: [256]u8 = undefined;
    const embed = try r.tableAt(try r.name(&nb, "embed_tokens", .{}), c.hidden_size);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try m.owned.own(final_norm);
    return .{
        .embed = .{ .table = embed, .hidden = @intCast(c.hidden_size), .act = .bfloat16 },
        .layers = layers,
        .final_norm = final_norm,
        .head = try r.head(&m.owned, embed, c.hidden_size, c.tie_word_embeddings),
    };
}

fn modeOf(q: abi.QuantMode) QuantMode {
    return switch (q) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => .affine,
    };
}

/// A routed-expert bank as the MoE block reads it: a dense bank resolves to the pack's declared scheme, as the engine's
/// `quantParamsHinted` does.
pub fn bankOf(l: Linear, q: abi.Quant) b.moe.Bank {
    const qp: QuantParams = if (l.s.ctx == null) .{ .bits = q.bits, .group_size = if (q.bits == 0) 64 else q.group_size, .mode = modeOf(q.mode) } else .{ .bits = l.bits, .group_size = l.group_size, .mode = l.mode };
    return .{ .w = l.w, .s = l.s, .b = l.b, .qp = qp };
}

/// One-expert views of affine banks (`moe_affine4.bindViews`), owned by `m`.
pub fn bankViews(o: *ld.Owned, gate: Linear, up: Linear, down: Linear, s: mlx.mlx_stream) !?moe_affine4.Views {
    var views: std.ArrayList(Array) = .empty;
    defer views.deinit(gpa);
    const Bk = moe_affine4.Bank;
    const v = try moe_affine4.bindViews(Bk{ .w = gate.w, .s = gate.s, .b = gate.b }, Bk{ .w = up.w, .s = up.s, .b = up.b }, Bk{ .w = down.w, .s = down.s, .b = down.b }, &views, gpa, s);
    for (views.items) |a| try o.own(a);
    return v;
}

fn loadMoe(m: *Loaded, r: *ld.Reader, c: *const config.Config, s: mlx.mlx_stream) !void {
    const e = &m.body.moe;
    e.layers = try gpa.alloc(Moe.Layer, c.num_hidden_layers);
    e.routes = try gpa.alloc(compiled.Closure, c.num_hidden_layers);
    @memset(e.routes, .{ .ctx = null });
    const hidden = c.hidden_size;
    const q_rows = c.num_attention_heads * c.head_dim;
    const n: u32 = c.num_hidden_layers;
    const freqs = try ropeFreqs(m, c);
    const mscale: f32 = @floatCast(rope_scaling.mscale(c.rope));
    const mscale_table: Array = if (mscale == 1.0) .{} else try scaled.mscaleTable(mscale, c.head_dim, c.head_dim, .bfloat16, s);
    if (mscale_table.ctx != null) try m.owned.own(mscale_table);
    var nb: [256]u8 = undefined;
    for (e.layers, 0..) |*L, i| {
        const li: u32 = @intCast(i);
        const q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
        const k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
        const v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
        const o = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), q_rows);
        for ([_]Linear{ q, k, v, o }) |l| try m.owned.linear(l);
        const router = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate", .{i}), hidden);
        const gate = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.gate_proj", .{i}), hidden);
        const up = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.up_proj", .{i}), hidden);
        const down = try r.bank(try r.name(&nb, "layers.{d}.mlp.switch_mlp.down_proj", .{i}), c.moe_intermediate_size);
        for ([_]Linear{ router, gate, up, down }) |l| try m.owned.linear(l);
        L.* = .{
            .input_norm = try norm(m, r, "input_layernorm.weight", i),
            .post_norm = try norm(m, r, "post_attention_layernorm.weight", i),
            .mixer = .{
                .q = .{ .lin = q, .dq = try ld.sideCopy(&m.owned, q, li, n, s) },
                .k = .{ .lin = k, .dq = try ld.sideCopy(&m.owned, k, li, n, s) },
                .v = .{ .lin = v, .dq = try ld.sideCopy(&m.owned, v, li, n, s) },
                .o = .{ .lin = o, .dq = try ld.sideCopy(&m.owned, o, li, n, s) },
                .q_norm = try norm(m, r, "self_attn.q_norm.weight", i),
                .k_norm = try norm(m, r, "self_attn.k_norm.weight", i),
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar))),
                    .rope_dims = @intCast(c.head_dim),
                    .rope_base = c.rope_theta,
                    .yarn_freqs = freqs,
                    .yarn_mscale = mscale_table,
                    .yarn_factor = mscale,
                    .rope_scale = rope_scaling.positionScale(c.rope),
                    .ntk = rope_scaling.ntk(c.rope, c.max_position_embeddings),
                    .output_gate = false,
                },
                .angles = &e.angles,
            },
            .mlp = .{
                .router = router,
                .route = .softmax,
                .gate = bankOf(gate, c.quant),
                .up = bankOf(up, c.quant),
                .down = bankOf(down, c.quant),
                .bank_views = try bankViews(&m.owned, gate, up, down, s),
                .top_k = @intCast(c.num_experts_per_tok),
                .experts = @intCast(c.num_experts),
            },
        };
        // The engine compiles the routing chain only where a GPU runs it.
        if (!mlx.noGpuBackend() and mlx.streamIsGpu(s)) {
            e.routes[i] = b.moe.compileRouting(&L.mlp, s);
            L.mlp.route_fn = e.routes[i];
        }
    }
    e.base.weights = try trunk(Moe, m, r, c, e.layers);
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (!abi.declaresMode(m.caps, f.mode)) return Verdict.refuse(.unsupported, "forward mode not declared");
    return switch (m.body) {
        .dense => |*d| run_mod.tableForward(Dense, &d.base, f, out),
        .moe => |*e| run_mod.tableForward(Moe, &e.base, f, out),
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
    var seed: u64 = 3;
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
    const head_ones = try a.alloc(f32, hd);
    @memset(head_ones, 1);
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
        try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.q_norm.weight"), .shape = &.{hd}, .data = head_ones });
        try tensors.append(a, .{ .name = try L.n(a, i, "self_attn.k_norm.weight"), .shape = &.{hd}, .data = head_ones });
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
        \\{"model_type":"qwen3","hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":2,
        \\ "num_key_value_heads":1,"head_dim":32,"vocab_size":64,"rms_norm_eps":1e-6,"rope_theta":1000000.0,
        \\ "max_position_embeddings":4096,"hidden_act":"silu","eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try sdk.fixture.safetensors(a, tensors.items) },
    }) };
}

// ── tests ──

test {
    _ = config;
}
