//! What Laguna, Flint and Kolibri share: an MLP slot holding a dense MLP or the routed experts per layer, and the load
//! of the family's checkpoints (mlx-lm / rec_mlx / Aleph Alpha layouts) over the engine's loaded tensors into a
//! `Decoder` whose mixer is `LagunaAttention`. The forward and the decode match `forwardMoeWith` /
//! `forwardMoeBatchedDecode` (`proof_test.zig`).
//! copied_from: sdk/bases/block.zig (`Either`, as the MLP-kind `Mlp`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const block = @import("../../sdk/bases/block.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const computeYarnFreqs = @import("../../sdk/blocks/rope/yarn.zig").computeYarnFreqs;
const DecodeCopies = @import("../../sdk/bases/blocks/decode_copies.zig").DecodeCopies;
const loader = @import("../llama/load.zig");
const config = @import("config.zig");

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Moe = b.moe.Moe;
const gpa = std.heap.c_allocator;

/// A dense MLP (`mlp_only_layers`) or the routed experts, per layer: a switch, never an indirect call (`Either`'s
/// kind is the mixer's).
pub const Mlp = struct {
    pub const kind: block.Kind = .mlp;
    pub const supports_batched_rows = b.mlp.GatedMlp.supports_batched_rows and Moe.supports_batched_rows;
    pub const Weights = union(enum) { dense: b.mlp.GatedMlp.Weights, moe: Moe.Weights };

    pub fn forward(w: *const Weights, c: *block.Ctx, x: ops.Array) !ops.Array {
        return switch (w.*) {
            .dense => |*d| b.mlp.GatedMlp.forward(d, c, x),
            .moe => |*m| Moe.forward(m, c, x),
        };
    }
};

/// The model a family plugin loads: the base's weights plus everything it owns.
pub fn Loaded(comptime D: type) type {
    return struct {
        const Self = @This();
        base: bases.Model(D),
        layers: []D.Layer = &.{},
        owned: loader.Owned = .{},
        route_fns: []compiled.Closure = &.{},

        pub fn release(m: *Self) void {
            for (m.route_fns) |*f| compiled.free(f);
            gpa.free(m.route_fns);
            m.owned.release();
            gpa.free(m.layers);
            m.base.shared.deinit();
        }
    };
}

/// The model's YaRN tables, owned by `o`: the f32 denominators `mlx_fast_rope` divides by and the f32 `[head_dim]`
/// attention factor on the rotated dims (1 on the tail), as the engine's init builds them.
fn yarnTables(o: *loader.Owned, c: *const config.Config) !struct { freqs: Array, mscale: Array } {
    const y = c.yarn.?;
    const rotary = c.yarnRotaryDim();
    const half: usize = rotary / 2;
    const f64s = try gpa.alloc(f64, half);
    defer gpa.free(f64s);
    computeYarnFreqs(f64s, c.head_dim, c.partial_rotary_factor_global, c.rope_theta, y.factor, y.beta_fast, y.beta_slow, y.orig_max, true);
    const f32s = try gpa.alloc(f32, @max(half, c.head_dim));
    defer gpa.free(f32s);
    for (f64s, 0..) |v, i| f32s[i] = @floatCast(v);
    const freqs = mlx.mlx_array_new_data(f32s.ptr, &[_]c_int{@intCast(half)}, 1, .float32);
    try o.own(freqs);
    for (f32s[0..c.head_dim], 0..) |*m, i| m.* = if (i < rotary) y.attention_factor else 1.0;
    const mscale = mlx.mlx_array_new_data(f32s.ptr, &[_]c_int{@intCast(c.head_dim)}, 1, .float32);
    try o.own(mscale);
    return .{ .freqs = freqs, .mscale = mscale };
}

/// `Decoder(TableEmbed, RmsNorm, Mixer, Mlp, AddNorm, LmHead)` loaded from the family's checkpoint; `Mixer` is
/// `DecodeCopies(LagunaAttention(...))` or kolibri1's `PostNormed` of it, and `D`'s MLP `Mlp` or kolibri1's `PostNormed(Moe)`.
pub fn loadFamily(comptime D: type, ctx: *const abi.LoadCtx) !*Loaded(D) {
    const c = config.of(ctx.parsed);
    var r = try loader.Reader.open(ctx, c.block_json, &.{"model"}, "embed_tokens.weight");
    defer r.close();
    const s = r.s;
    const kolibri = c.kind == .kolibri1;

    const M = Loaded(D);
    const m = try gpa.create(M);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = s } };
    errdefer m.release();
    m.layers = try gpa.alloc(D.Layer, c.num_hidden_layers);
    m.route_fns = try gpa.alloc(compiled.Closure, c.num_hidden_layers);
    @memset(m.route_fns, .{ .ctx = null });
    const o = &m.owned;
    const yarn = if (c.yarn != null) try yarnTables(o, c) else null;

    const hidden = c.hidden_size;
    var nb: [256]u8 = undefined;
    for (m.layers, 0..) |*L, i| {
        const heads = c.heads[i];
        const full = c.global[i];
        const q = try r.proj(try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden);
        try o.linear(q);
        const k = try r.proj(try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden);
        try o.linear(k);
        const v = try r.proj(try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden);
        try o.linear(v);
        const ow = try r.proj(try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), heads * c.head_dim);
        try o.linear(ow);
        const qn = try r.need(try r.name(&nb, "layers.{d}.self_attn.q_norm.weight", .{i}));
        try o.own(qn);
        const kn = try r.need(try r.name(&nb, "layers.{d}.self_attn.k_norm.weight", .{i}));
        try o.own(kn);
        const yl = full and yarn != null;
        const attn: b.laguna_attention.LagunaAttention(.{}).Weights = .{
            .q = q,
            .k = k,
            .v = v,
            .o = ow,
            .q_norm = qn,
            .k_norm = kn,
            .geo = .{
                .heads = @intCast(heads),
                .kv_heads = @intCast(c.num_key_value_heads),
                .head_dim = @intCast(c.head_dim),
                .scale = c.scale,
                .rope_dims = @intCast(c.ropeDims(i)),
                .rope_base = if (full) c.rope_theta else c.rope_local_base_freq,
                .rope_scale = 1.0,
                .nope = c.no_rope[i],
                .freqs = if (yl) yarn.?.freqs else .{},
            },
            .mscale = if (yl) yarn.?.mscale else .{},
            .window = if (full) 0 else c.sliding_window,
        };
        var dqs: [4]Linear = undefined;
        for (&dqs, [_]Linear{ q, k, v, ow }) |*d, l| {
            d.* = try b.decode_copies.decodeCopy(l, @intCast(i), c.num_hidden_layers, s);
            try o.linear(d.*);
        }
        L.mixer = try mixerOf(D, &r, o, c, i, attn, dqs);
        L.input_norm = try r.need(try r.name(&nb, "layers.{d}.input_layernorm.weight", .{i}));
        try o.own(L.input_norm);
        // Kolibri's checkpoint names the pre-FFN norm `post_attention_layernorm` and the attention output's `post_attn_norm`.
        L.post_norm = try r.need(try r.name(&nb, "layers.{d}.post_attention_layernorm.weight", .{i}));
        try o.own(L.post_norm);

        const moe_layer = if (kolibri) true else r.has(try r.name(&nb, "layers.{d}.mlp.switch_mlp.gate_proj.weight", .{i}));
        if (!moe_layer) {
            var gate = try r.proj(try r.name(&nb, "layers.{d}.mlp.gate_proj", .{i}), hidden);
            var up = try r.proj(try r.name(&nb, "layers.{d}.mlp.up_proj", .{i}), hidden);
            const down = try r.proj(try r.name(&nb, "layers.{d}.mlp.down_proj", .{i}), c.intermediate_size);
            const gu = if (gate.s.ctx != null) try loader.join(o, &.{ &gate, &up }, s) else b.linear.Joined{};
            for ([_]Linear{ gate, up, down }) |l| try o.linear(l);
            L.mlp = try mlpOf(D, &r, o, i, .{ .dense = .{ .gate = gate, .up = up, .down = down, .gu = gu } });
            continue;
        }
        var w = try moeOf(&r, o, c, i);
        m.route_fns[i] = b.moe.compileRouting(&w, s);
        w.route_fn = m.route_fns[i];
        L.mlp = try mlpOf(D, &r, o, i, .{ .moe = w });
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

/// The attention weights in `D`'s mixer: the per-type extras (Laguna's softplus gate, Flint's sinks in the activation
/// dtype) and Kolibri's output norm.
fn mixerOf(comptime D: type, r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: usize, a: anytype, dqs: [4]Linear) !@FieldType(D.Layer, "mixer") {
    const W = @FieldType(D.Layer, "mixer");
    if (!@hasField(W, "norm")) return attnOf(W, r, o, c, i, a, dqs);
    var nb: [256]u8 = undefined;
    const n = try r.need(try r.name(&nb, "layers.{d}.post_attn_norm.weight", .{i}));
    try o.own(n);
    return .{ .inner = try attnOf(@FieldType(W, "inner"), r, o, c, i, a, dqs), .norm = n };
}

/// A `DecodeCopies(LagunaAttention(...))`'s weights: `a` plus Laguna's softplus gate or Flint's sinks (in the
/// activation dtype), and the side copies.
fn attnOf(comptime W: type, r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: usize, a: anytype, dqs: [4]Linear) !W {
    var nb: [256]u8 = undefined;
    var w = copyAttn(@FieldType(W, "inner"), a);
    switch (c.kind) {
        .laguna => {
            w.gate = try r.proj(try r.name(&nb, "layers.{d}.self_attn.g_proj", .{i}), c.hidden_size);
            try o.linear(w.gate);
        },
        .flint => {
            const raw = try r.need(try r.name(&nb, "layers.{d}.self_attn.softmax_offset", .{i}));
            if (mlx.mlx_array_dtype(raw) == .bfloat16) {
                try o.own(raw);
                w.sinks = raw;
            } else {
                defer _ = mlx.mlx_array_free(raw);
                var cast = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(cast);
                try mlx.check(mlx.mlx_astype(&cast, raw, .bfloat16, r.s));
                try mlx.check(mlx.mlx_array_eval(cast));
                try o.own(cast);
                w.sinks = cast;
            }
        },
        .kolibri1 => {},
    }
    return .{ .inner = w, .dq = dqs };
}

/// `a` (any `LagunaAttention(...)` weights) as `W`'s; the extras stay unset.
fn copyAttn(comptime W: type, a: anytype) W {
    return .{ .q = a.q, .k = a.k, .v = a.v, .o = a.o, .q_norm = a.q_norm, .k_norm = a.k_norm, .geo = a.geo, .mscale = a.mscale, .window = a.window };
}

/// The MLP slot of `D`: the family's `Mlp`, or Kolibri's routed experts behind their output norm.
fn mlpOf(comptime D: type, r: *loader.Reader, o: *loader.Owned, i: usize, w: Mlp.Weights) !@FieldType(D.Layer, "mlp") {
    const W = @FieldType(D.Layer, "mlp");
    if (W == Mlp.Weights) return w;
    var nb: [256]u8 = undefined;
    const n = try r.need(try r.name(&nb, "layers.{d}.post_ffn_norm.weight", .{i}));
    try o.own(n);
    return .{ .inner = w.moe, .norm = n };
}

/// The routed experts: Laguna/Flint read qwen3_moe's names (`mlp.gate` or HF-native `mlp.gate.proj` router,
/// `mlp.switch_mlp.*`, an ungated `mlp.shared_expert.*`, the selection bias zero where a converter dropped it);
/// Kolibri reads `mlp.experts.*` (or `mlp.switch_mlp.*`), a dense router widened to f32, `mlp.expert_bias` and
/// `mlp.shared_experts.*`.
fn moeOf(r: *loader.Reader, o: *loader.Owned, c: *const config.Config, i: usize) !Moe.Weights {
    const hidden = c.hidden_size;
    const kolibri = c.kind == .kolibri1;
    var nb: [256]u8 = undefined;
    var kb: [256]u8 = undefined;
    const router_base = if (kolibri or r.has(try r.name(&nb, "layers.{d}.mlp.gate.weight", .{i})) or
        !r.has(try r.name(&kb, "layers.{d}.mlp.gate.proj.weight", .{i}))) "mlp.gate" else "mlp.gate.proj";
    var router = try r.proj(try r.name(&nb, "layers.{d}.{s}", .{ i, router_base }), hidden);
    try o.linear(router);
    const container = if (!kolibri or !r.has(try r.name(&nb, "layers.{d}.mlp.experts.gate_proj.weight", .{i}))) "mlp.switch_mlp" else "mlp.experts";
    const banks = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var bank: [3]b.moe.Bank = undefined;
    for (banks, &bank, 0..) |nm, *bk, j| {
        bk.* = try r.bank(try r.name(&nb, "layers.{d}.{s}.{s}", .{ i, container, nm }), if (j == 2) c.moe_intermediate_size else hidden);
        for ([_]Array{ bk.w, bk.s, bk.b }) |a| try o.own(a);
    }
    const shared_base = if (kolibri) "mlp.shared_experts" else "mlp.shared_expert";
    var shared: ?b.moe.SharedExpert = null;
    if (kolibri or r.has(try r.name(&nb, "layers.{d}.{s}.gate_proj.weight", .{ i, shared_base }))) {
        const gate = try r.proj(try r.name(&nb, "layers.{d}.{s}.gate_proj", .{ i, shared_base }), hidden);
        try o.linear(gate);
        const up = try r.proj(try r.name(&nb, "layers.{d}.{s}.up_proj", .{ i, shared_base }), hidden);
        try o.linear(up);
        // The shared width is the up projection's output rows (dense weights load as [in, out]).
        const ush = mlx.getShape(up.w);
        const shared_inter: u32 = @intCast(if (up.s.ctx == null) ush[1] else ush[0]);
        const down = try r.proj(try r.name(&nb, "layers.{d}.{s}.down_proj", .{ i, shared_base }), shared_inter);
        try o.linear(down);
        shared = .{ .gate = gate, .up = up, .down = down };
    }
    var w: Moe.Weights = .{
        .router = router,
        .route = undefined,
        .gate = bank[0],
        .up = bank[1],
        .down = bank[2],
        .shared = shared,
        .top_k = @intCast(c.top_k),
        .experts = @intCast(c.num_experts),
    };
    if (kolibri) {
        if (router.s.ctx != null) return error.UnsupportedKolibriConfig;
        var r32 = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(r32);
        try mlx.check(mlx.mlx_astype(&r32, router.w, .float32, r.s));
        try o.own(r32);
        router.w = r32;
        w.router = router;
        const bias = try r.need(try r.name(&nb, "layers.{d}.mlp.expert_bias", .{i}));
        try o.own(bias);
        w.route = .{ .logit_bias = bias };
        return w;
    }
    const bias = blk: {
        if (try optional(r, try r.name(&nb, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}))) |a| break :blk a;
        if (c.kind == .flint) if (try optional(r, try r.name(&nb, "layers.{d}.mlp.gate.expert_bias", .{i}))) |a| break :blk a;
        // The reference zero-inits the selection bias; some conversions drop it.
        var zeros = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(zeros);
        try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{@intCast(c.num_experts)}, 1, .float32, r.s));
        break :blk zeros;
    };
    try o.own(bias);
    w.route = .{ .sigmoid_bias = .{ .expert_bias = bias, .norm = c.route_norm, .scale = c.route_scale } };
    return w;
}

fn optional(r: *const loader.Reader, full: []const u8) !?Array {
    return if (r.has(full)) try r.need(full) else null;
}

/// The table's unload and forward over a `Loaded(D)`; `batches` = the type's `Caps.batches_decode` (the legacy
/// decodes Laguna and Kolibri serially, so their tables refuse batched rows).
pub fn Table(comptime D: type, comptime batches: bool) type {
    if (batches and !D.batches_decode) @compileError("a batched family type over a base that cannot batch");
    return struct {
        pub fn unload(mp: *anyopaque) callconv(.c) void {
            const m: *Loaded(D) = @ptrCast(@alignCast(mp));
            m.release();
            gpa.destroy(m);
        }

        pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            if (!batches and f.mode == .batched_rows) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
            const m: *Loaded(D) = @ptrCast(@alignCast(mp));
            return run_mod.tableForward(D, &m.base, f, out);
        }

        pub fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
            return loadFamily(D, ctx) catch |e| {
                out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
                return null;
            };
        }

        /// Every term is the engine's default arithmetic over the shell.
        pub fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
            out.* = .{};
        }
    };
}

/// The conformance pack of a family plugin: two layers at width 64 (2 heads of 32, one KV head, 4 experts of 32,
/// top 2), seeded random f32; Laguna and Flint make layer 0 dense. `json` carries those dimensions.
pub fn conformancePack(a: std.mem.Allocator, comptime kind: config.Kind, json: []const u8) !@import("../../sdk/root.zig").fixture.Fixture {
    const fx = @import("../../sdk/root.zig").fixture;
    const V = 64;
    const H = 64;
    const HD = 32;
    const HEADS = 2;
    const INTER = 128;
    const E = 4;
    const MI = 32;
    var tensors: std.ArrayList(fx.Tensor) = .empty;
    var seed: u64 = 11;
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
        fn one(p: @This(), name: []const u8, data: []const f32) !void {
            try p.list.append(p.a, .{ .name = name, .shape = try p.a.dupe(u64, &.{data.len}), .data = data });
        }
        fn at(p: @This(), li: usize, leaf: []const u8) ![]const u8 {
            return std.fmt.allocPrint(p.a, "model.layers.{d}.{s}", .{ li, leaf });
        }
    };
    const p: Put = .{ .a = a, .list = &tensors, .seed = &seed };
    try p.rand("model.embed_tokens.weight", &.{ V, H }, 0.5);
    try p.rand("lm_head.weight", &.{ V, H }, 0.5);
    try p.one("model.norm.weight", ones);
    for (0..2) |i| {
        try p.one(try p.at(i, "input_layernorm.weight"), ones);
        try p.one(try p.at(i, "post_attention_layernorm.weight"), ones);
        try p.one(try p.at(i, "self_attn.q_norm.weight"), ones[0..HD]);
        try p.one(try p.at(i, "self_attn.k_norm.weight"), ones[0..HD]);
        try p.rand(try p.at(i, "self_attn.q_proj.weight"), &.{ HEADS * HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.k_proj.weight"), &.{ HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.v_proj.weight"), &.{ HD, H }, 0.1);
        try p.rand(try p.at(i, "self_attn.o_proj.weight"), &.{ H, HEADS * HD }, 0.1);
        switch (kind) {
            .laguna => try p.rand(try p.at(i, "self_attn.g_proj.weight"), &.{ HEADS, H }, 0.1),
            .flint => try p.rand(try p.at(i, "self_attn.softmax_offset"), &.{HEADS}, 1.0),
            .kolibri1 => {
                try p.one(try p.at(i, "post_attn_norm.weight"), ones);
                try p.one(try p.at(i, "post_ffn_norm.weight"), ones);
            },
        }
        if (i == 0 and kind != .kolibri1) {
            try p.rand(try p.at(i, "mlp.gate_proj.weight"), &.{ INTER, H }, 0.1);
            try p.rand(try p.at(i, "mlp.up_proj.weight"), &.{ INTER, H }, 0.1);
            try p.rand(try p.at(i, "mlp.down_proj.weight"), &.{ H, INTER }, 0.1);
            continue;
        }
        const ex = if (kind == .kolibri1) "mlp.experts" else "mlp.switch_mlp";
        const se = if (kind == .kolibri1) "mlp.shared_experts" else "mlp.shared_expert";
        try p.rand(try p.at(i, "mlp.gate.weight"), &.{ E, H }, 0.1);
        try p.rand(try p.at(i, if (kind == .kolibri1) "mlp.expert_bias" else "mlp.gate.e_score_correction_bias"), &.{E}, 0.1);
        for ([_][]const u8{ "gate_proj", "up_proj" }) |nm| {
            try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.{s}.weight", .{ ex, nm })), &.{ E, MI, H }, 0.1);
            try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.{s}.weight", .{ se, nm })), &.{ MI, H }, 0.1);
        }
        try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.down_proj.weight", .{ex})), &.{ E, H, MI }, 0.1);
        try p.rand(try p.at(i, try std.fmt.allocPrint(a, "{s}.down_proj.weight", .{se})), &.{ H, MI }, 0.1);
    }
    return .{ .files = try a.dupe(fx.File, &.{
        .{ .name = "config.json", .bytes = json },
        .{ .name = "model.safetensors", .bytes = try fx.safetensors(a, tensors.items) },
    }) };
}
