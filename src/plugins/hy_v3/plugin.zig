//! Tencent Hunyuan 3 (`hy_v3`): the decoder base over the engine's loaded tensors. Full attention with per-head q/k
//! RMS norms and no output gate (decode reads the requantized side copies of dense projections), a dense SwiGLU below
//! `first_k_dense_replace`, then sigmoid-routed experts with a selection-only bias and an ungated shared expert,
//! closing on add_norm seams. Decodes serially, as the engine always served it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const gated = @import("../../sdk/bases/blocks/gated_attention.zig");
const qk_rope = @import("../../sdk/bases/blocks/qk_rope.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const reader = @import("../../sdk/bases/reader.zig");
const EitherMlp = @import("../lfm2/plugin.zig").EitherMlp;
const mtp_head = @import("mtp_head.zig");
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
    .mixer = gated.GatedAttention,
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
    .cache = &.{},
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded), .head = head_hooks },
    .weights = &.{},
};

/// Every layer full attention at the shell's geometry: the engine's default arithmetic.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},
    angles: qk_rope.AngleCache = .{},
    /// The sigmoid routing chain compiled once for every MoE layer (the engine's `compileMoeRouting`).
    route_fn: compiled.Closure = .{ .ctx = null },
    /// The MTP head, once `HeadHooks.load` found one.
    head: ?*mtp_head.Head = null,

    fn release(m: *Loaded) void {
        if (m.head) |h| {
            h.deinit();
            gpa.destroy(h);
        }
        compiled.free(&m.route_fn);
        m.angles.deinit();
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
    const prefix = try reader.probePrefix(ctx, "model", &.{ "language_model.model", "model.language_model" }, "embed_tokens.weight");
    var r = try reader.Reader.open(ctx, c.block_json, prefix);
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| L.* = try loadLayer(m, &r, c, @intCast(i));

    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const embed = try r.table(try r.name(&nb, "embed_tokens", .{}), hidden);
    try m.owned.linear(embed);
    const final_norm = try r.need(try r.name(&nb, "norm.weight", .{}));
    try m.owned.own(final_norm);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = try headOf(&r, m, embed, hidden, c.tie_word_embeddings),
    };
    // The engine's ternary arm (qmv2's kernels) for a 2-bit group-128 pack whose every bias is -scale.
    if (c.quant.mode == .affine and c.quant.bits == 2 and c.quant.group_size == 128)
        try reader.armTernary(.{ &m.layers, &m.base.weights.head }, null, reader.gatherOnly(&m.base.weights.head, embed), r.s);
    return m;
}

/// A projection with its decode side copy.
fn proj(m: *Loaded, r: *reader.Reader, c: *const config.Config, base: []const u8, in_dim: u32, layer: u32) !gated.Proj {
    const lin = try r.proj(base, in_dim);
    try m.owned.linear(lin);
    const dq = try b.decode_copies.decodeCopy(lin, layer, c.num_hidden_layers, r.s);
    try m.owned.linear(dq);
    return .{ .lin = lin, .dq = dq };
}

fn loadLayer(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !Model.Layer {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    var norms: [4]Array = @splat(.{ .ctx = null });
    const leaves = [_][]const u8{ "input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm" };
    const n_norms: usize = if (c.qk_norm) 4 else 2;
    for (norms[0..n_norms], leaves[0..n_norms]) |*n, leaf| {
        n.* = try r.need(try r.name(&nb, "layers.{d}.{s}.weight", .{ i, leaf }));
        try m.owned.own(n.*);
    }
    const mixer: gated.GatedAttention.Weights = .{
        .q = try proj(m, r, c, try r.name(&nb, "layers.{d}.self_attn.q_proj", .{i}), hidden, i),
        .k = try proj(m, r, c, try r.name(&nb, "layers.{d}.self_attn.k_proj", .{i}), hidden, i),
        .v = try proj(m, r, c, try r.name(&nb, "layers.{d}.self_attn.v_proj", .{i}), hidden, i),
        .o = try proj(m, r, c, try r.name(&nb, "layers.{d}.self_attn.o_proj", .{i}), c.num_attention_heads * c.head_dim, i),
        .q_norm = norms[2],
        .k_norm = norms[3],
        .geo = .{
            .heads = @intCast(c.num_attention_heads),
            .kv_heads = @intCast(c.num_key_value_heads),
            .head_dim = @intCast(c.head_dim),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar))),
            .rope_dims = @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor),
            .rope_base = c.rope_theta,
            .output_gate = false,
        },
        .angles = &m.angles,
    };
    return .{ .input_norm = norms[0], .mixer = mixer, .post_norm = norms[1], .mlp = try mlpOf(m, r, c, i) };
}

fn mlpOf(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !Mlp.Weights {
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
    const router = try r.proj(try r.name(&nb, "layers.{d}.mlp.router.gate", .{i}), hidden);
    try m.owned.linear(router);
    // mlx-lm stacks the experts under `switch_mlp`, the ox-ox conversions under `experts`; `experts` wins.
    const container = if (r.has(try r.name(&nb, "layers.{d}.mlp.experts.gate_proj.weight", .{i}))) "experts" else "switch_mlp";
    var banks: [3]moe.Bank = undefined;
    for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, c.moe_intermediate }) |*bk, leaf, in_dim| {
        const l = try r.proj(try r.name(&nb, "layers.{d}.mlp.{s}.{s}", .{ i, container, leaf }), in_dim);
        try m.owned.linear(l);
        bk.* = reader.bank(l, c.quant);
    }
    // `moe_router_enable_expert_bias: false` ships none; the reference keeps zeros (plain sigmoid top-k).
    var bias = try r.share(try r.name(&nb, "layers.{d}.mlp.expert_bias", .{i}));
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
    };
    if (try r.projOpt(try r.name(&nb, "layers.{d}.mlp.shared_mlp.gate_proj", .{i}), hidden)) |sg| {
        try m.owned.linear(sg);
        const su = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_mlp.up_proj", .{i}), hidden);
        try m.owned.linear(su);
        const sd = try r.proj(try r.name(&nb, "layers.{d}.mlp.shared_mlp.down_proj", .{i}), c.shared_intermediate);
        try m.owned.linear(sd);
        w.shared = .{ .gate = sg, .up = su, .down = sd };
    }
    if (m.route_fn.ctx == null) m.route_fn = moe.compileRouting(&w, r.s);
    w.route_fn = m.route_fn;
    return .{ .second = w };
}

/// The head as the engine finds it: `lm_head` at the root, else the tied embedding.
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

const head_hooks = reader.headHooks(Loaded, mtp_head.Head, struct {
    fn load(ctx: *const abi.LoadCtx) !?*mtp_head.Head {
        return mtp_head.load(ctx, config.of(ctx.parsed));
    }
}.load, mtp_head.forward);

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    m.release();
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    // The blocks batch, but the engine decodes hy_v3 serially (caps): batched rows stay undeclared.
    if (f.mode == .batched_rows) return abi.Verdict.refuse(.unsupported, "forward mode not declared");
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}

/// The tiny pack the conformance suite runs: a dense layer and an MoE layer, width 64, head 32, seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    var t = sdk.tensors.Tensors.init(a, 1);
    try t.rand("model.embed_tokens.weight", .{}, &.{ 64, h }, .F32, 0.5);
    try t.fill("model.norm.weight", .{}, &.{h}, .F32, 1);
    try t.rand("lm_head.weight", .{}, &.{ 64, h }, .F32, 0.1);
    for (0..2) |i| {
        try t.fill("model.layers.{d}.input_layernorm.weight", .{i}, &.{h}, .F32, 1);
        try t.fill("model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}, .F32, 1);
        try t.fill("model.layers.{d}.self_attn.q_norm.weight", .{i}, &.{32}, .F32, 1);
        try t.fill("model.layers.{d}.self_attn.k_norm.weight", .{i}, &.{32}, .F32, 1);
        try t.rand("model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ 64, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
        try t.rand("model.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, 64 }, .F32, 0.1);
        if (i == 0) {
            try t.rand("model.layers.{d}.mlp.gate_proj.weight", .{i}, &.{ 128, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.up_proj.weight", .{i}, &.{ 128, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, 128 }, .F32, 0.1);
        } else {
            try t.rand("model.layers.{d}.mlp.router.gate.weight", .{i}, &.{ 4, h }, .F32, 0.3);
            try t.rand("model.layers.{d}.mlp.expert_bias", .{i}, &.{4}, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.gate_proj.weight", .{i}, &.{ 4, 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.up_proj.weight", .{i}, &.{ 4, 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.switch_mlp.down_proj.weight", .{i}, &.{ 4, h, 32 }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.shared_mlp.gate_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.shared_mlp.up_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
            try t.rand("model.layers.{d}.mlp.shared_mlp.down_proj.weight", .{i}, &.{ h, 32 }, .F32, 0.1);
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"hy_v3","vocab_size":64,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"rms_norm_eps":1e-5,"rope_theta":10000.0,
        \\ "max_position_embeddings":4096,"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":32,
        \\ "num_shared_experts":1,"router_scaling_factor":2.5,"first_k_dense_replace":1,"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
}
