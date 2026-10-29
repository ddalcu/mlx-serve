//! Nemotron-H (`nemotron_h`, `nemotron_h_puzzle`, the omni wrapper's text trunk): the hybrid base over the engine's
//! loaded tensors. One operation per layer (Mamba2, NoPE attention, the sigmoid-routed ReLU² MoE with its shared
//! expert or the Puzzle latent pair, a ReLU² MLP), closing on `add_norm` seams with the MoE combine folded in.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const reader = @import("../../sdk/bases/reader.zig");
const mtp_head = @import("mtp_head.zig");
const row_exact = @import("../../sdk/bases/blocks/row_exact.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const b = bases.blocks;
const Array = mlx.mlx_array;
const Linear = b.linear.Linear;
const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

pub const Mixer = bases.Either(b.mamba2.Mamba2, b.attention.HybridAttention);
pub const Ffn = bases.Either(b.nemotron_moe.NemotronMoe, b.mlp.ReluSq);
pub const Op = bases.Either(Mixer, Ffn);

pub const Model = bases.Hybrid(.{
    .embed = b.embed.TableEmbed,
    .norm = b.norm.RmsNorm,
    .op = Op,
    .mlp = b.mlp.GatedMlp,
    .seam = b.seam.AddNorm,
    .head = b.head.LmHead,
});

comptime {
    if (Model.batches_decode != config.batches_decode) @compileError("nemotron_h: config.batches_decode disagrees with the base");
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
    .cache = &.{ .layout = config.layout },
    .spec = &.{ .trunk_weight = b.trunk_weight.trunkWeightHook(Loaded), .head = head_hooks },
    .weights = &.{},
};

/// Every term is the engine's default arithmetic over the shell and the layout.
fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub const Loaded = struct {
    base: bases.Model(Model),
    layers: []Model.Layer = &.{},
    owned: reader.Owned = .{},
    /// The MTP head, once `HeadHooks.load` found one.
    head: ?*mtp_head.Head = null,

    fn release(m: *Loaded) void {
        if (m.head) |h| {
            h.deinit();
            gpa.destroy(h);
        }
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
    // The omni wrapper nests the trunk under `language_model`; the checkpoint decides.
    const prefix = try reader.probePrefix(ctx, "backbone", &.{"language_model.backbone"}, "embeddings.weight");
    var r = try reader.Reader.open(ctx, c.block_json, prefix);
    defer r.close();
    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .base = .{ .weights = undefined, .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps), .a = gpa, .s = r.s, .placeholders = ctx.parsed.placeholder_ids } };
    errdefer m.release();
    m.layers = try gpa.alloc(Model.Layer, c.num_hidden_layers);
    for (m.layers, 0..) |*L, i| {
        var nb: [256]u8 = undefined;
        const norm = try r.need(try r.name(&nb, "layers.{d}.norm.weight", .{i}));
        try m.owned.own(norm);
        L.* = .{ .input_norm = norm, .op = try opOf(m, &r, c, @intCast(i)) };
    }

    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const embed = try r.table(try r.name(&nb, "embeddings", .{}), hidden);
    try m.owned.gather(embed);
    const final_norm = try r.need(try r.name(&nb, "norm_f.weight", .{}));
    try m.owned.own(final_norm);
    const head = try headOf(&r, m, embed, hidden, c.tie_word_embeddings);
    m.base.weights = .{
        .embed = .{ .table = embed, .hidden = @intCast(hidden), .act = .bfloat16 },
        .layers = m.layers,
        .final_norm = final_norm,
        .head = head,
    };
    // The engine's ternary arm (qmv2's kernels) for a 2-bit group-128 pack whose every bias is -scale; its MoE
    // (router, experts, shared and latent projections) runs outside `qmatmul`, on stock kernels.
    if (c.quant.mode == .affine and c.quant.bits == 2 and c.quant.group_size == 128)
        try reader.armTernary(.{ &m.layers, &m.base.weights.head }, b.nemotron_moe.NemotronMoe.Weights, reader.gatherOnly(&m.base.weights.head, embed), r.s);
    // A bound drafter's verify rows reproduce serial decode only when every projection fits the row kernels.
    m.base.row_exact = if (m.owned.outside_rows or !row_exact.fits(&head)) .off else .rows;
    return m;
}

/// The axis-`axis` width of the source's `full` tensor (a bank's width, read before any transpose).
fn widthOf(r: *const reader.Reader, full: []const u8, axis: usize) !u32 {
    const a = r.src.tensor(full);
    if (a.ctx == null) return error.MissingWeight;
    return @intCast(ops.shape(.{ .ctx = a.ctx })[axis]);
}

fn opOf(m: *Loaded, r: *reader.Reader, c: *const config.Config, i: u32) !Op.Weights {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    switch (c.layers[i]) {
        .mamba2 => {
            const mc = c.mamba;
            const in_proj = try r.proj(try r.name(&nb, "layers.{d}.mixer.in_proj", .{i}), hidden);
            try m.owned.linear(in_proj);
            const out_proj = try r.proj(try r.name(&nb, "layers.{d}.mixer.out_proj", .{i}), mc.heads * mc.head_dim);
            try m.owned.linear(out_proj);
            var raw: [6]Array = undefined;
            for (&raw, [_][]const u8{ "conv1d.weight", "conv1d.bias", "A_log", "D", "dt_bias", "norm.weight" }, 0..) |*a, leaf, k| {
                const key = try r.name(&nb, "layers.{d}.mixer.{s}", .{ i, leaf });
                a.* = if (k == 1) try r.share(key) else try r.need(key);
                try m.owned.own(a.*);
            }
            return .{ .first = .{ .first = .{
                .in_proj = in_proj,
                .conv_w = raw[0],
                .conv_b = raw[1],
                .A_log = raw[2],
                .D = raw[3],
                .dt_bias = raw[4],
                .norm = raw[5],
                .out_proj = out_proj,
                .geo = .{
                    .heads = @intCast(mc.heads),
                    .head_dim = @intCast(mc.head_dim),
                    .groups = @intCast(mc.groups),
                    .state = @intCast(mc.state),
                    .kernel = @intCast(mc.kernel),
                    .dt_min = mc.dt_min,
                    .dt_max = mc.dt_max,
                },
            } } };
        },
        .attention => {
            var p: [4]Linear = undefined;
            for (&p, [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, 0..) |*l, leaf, k| {
                l.* = try r.proj(try r.name(&nb, "layers.{d}.mixer.{s}", .{ i, leaf }), if (k == 3) c.num_attention_heads * c.head_dim else hidden);
                try m.owned.linear(l.*);
            }
            return .{ .first = .{ .second = .{
                .q = p[0],
                .k = p[1],
                .v = p[2],
                .o = p[3],
                .geo = .{
                    .heads = @intCast(c.num_attention_heads),
                    .kv_heads = @intCast(c.num_key_value_heads),
                    .head_dim = @intCast(c.head_dim),
                    .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))),
                    .rope_dims = @intCast(c.head_dim),
                    .rope_base = c.rope_theta,
                    .rope_scale = 1.0,
                    .nope = true,
                },
                .keep = c.max_position_embeddings,
            } } };
        },
        .moe => return .{ .second = .{ .first = try moeOf(&m.owned, r, c, try std.fmt.bufPrint(&nb, "layers.{d}", .{i})) } },
        .mlp => {
            const up = try r.proj(try r.name(&nb, "layers.{d}.mixer.up_proj", .{i}), hidden);
            try m.owned.linear(up);
            const down = try r.proj(try r.name(&nb, "layers.{d}.mixer.down_proj", .{i}), c.intermediate_size);
            try m.owned.linear(down);
            return .{ .second = .{ .second = .{ .up = up, .down = down } } };
        },
    }
}

/// The MoE under `<base>.mixer` (`layers.{i}` in the trunk; the MTP head's own block).
pub fn moeOf(own: *reader.Owned, r: *reader.Reader, c: *const config.Config, base: []const u8) !b.nemotron_moe.NemotronMoe.Weights {
    var nb: [256]u8 = undefined;
    const hidden = c.hidden_size;
    const latent = r.has(try r.name(&nb, "{s}.mixer.fc1_latent_proj.weight", .{base}));
    if (latent != r.has(try r.name(&nb, "{s}.mixer.fc2_latent_proj.weight", .{base}))) return error.MissingWeight;
    if (c.moe.latent > 0 and !latent) return error.MissingWeight;
    // Each layer's banks carry their own widths (Puzzle layers differ): read them off the tensors.
    const ein: u32 = if (latent) try widthOf(r, try r.name(&nb, "{s}.mixer.fc1_latent_proj.weight", .{base}), 0) else hidden;
    const inter = try widthOf(r, try r.name(&nb, "{s}.mixer.switch_mlp.fc1.weight", .{base}), 1);
    var w: b.nemotron_moe.NemotronMoe.Weights = .{
        .router = try r.proj(try r.name(&nb, "{s}.mixer.gate", .{base}), hidden),
        // A dense router also keeps its stored `[experts, hidden]` rows for the row-exact router.
        .router_rows = if (r.has(try r.name(&nb, "{s}.mixer.gate.scales", .{base}))) .{} else try r.share(try r.name(&nb, "{s}.mixer.gate.weight", .{base})),
        .expert_bias = try r.need(try r.name(&nb, "{s}.mixer.gate.e_score_correction_bias", .{base})),
        .fc1 = try r.proj(try r.name(&nb, "{s}.mixer.switch_mlp.fc1", .{base}), ein),
        .fc2 = try r.proj(try r.name(&nb, "{s}.mixer.switch_mlp.fc2", .{base}), inter),
        .routing = .{
            .k = @intCast(c.moe.top_k),
            .n_group = @intCast(c.moe.n_group),
            .topk_group = @intCast(c.moe.topk_group),
            .renorm = c.moe.norm,
            .scale = c.moe.scale,
        },
    };
    try own.own(w.expert_bias);
    try own.own(w.router_rows);
    for ([_]Linear{ w.router, w.fc1, w.fc2 }) |l| try own.linear(l);
    if (try r.projOpt(try r.name(&nb, "{s}.mixer.shared_experts.up_proj", .{base}), hidden)) |up| {
        try own.linear(up);
        w.shared_up = up;
        const shared_inter = try widthOf(r, try r.name(&nb, "{s}.mixer.shared_experts.up_proj.weight", .{base}), 0);
        w.shared_down = try r.proj(try r.name(&nb, "{s}.mixer.shared_experts.down_proj", .{base}), shared_inter);
        try own.linear(w.shared_down);
    }
    if (latent) {
        w.latent_in = try r.proj(try r.name(&nb, "{s}.mixer.fc1_latent_proj", .{base}), hidden);
        try own.linear(w.latent_in);
        w.latent_out = try r.proj(try r.name(&nb, "{s}.mixer.fc2_latent_proj", .{base}), ein);
        try own.linear(w.latent_out);
    }
    return w;
}

/// The head as the engine finds it: `lm_head` beside the trunk (`language_model.lm_head` under the omni wrapper),
/// then at the root, then the tied embedding.
fn headOf(r: *reader.Reader, m: *Loaded, embed: Linear, hidden: u32, tied: bool) !Linear {
    var nb: [160]u8 = undefined;
    const nested = if (std.mem.eql(u8, r.prefix, "language_model.backbone")) "language_model" else r.prefix;
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
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return run_mod.tableForward(Model, &m.base, f, out);
}
/// The tiny pack the conformance suite runs: Mamba2, attention, MoE and MLP layers, width 64, seeded f32 weights.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    var t = sdk.tensors.Tensors.init(a, 1);
    try t.rand("backbone.embeddings.weight", .{}, &.{ 64, h }, .F32, 0.5);
    try t.fill("backbone.norm_f.weight", .{}, &.{h}, .F32, 1);
    try t.rand("lm_head.weight", .{}, &.{ 64, h }, .F32, 0.1);
    for ("M*E-", 0..) |kind, i| {
        try t.fill("backbone.layers.{d}.norm.weight", .{i}, &.{h}, .F32, 1);
        switch (kind) {
            // 2 heads x 32, 1 group, state 16: conv dim 64 + 32.
            'M' => {
                try t.rand("backbone.layers.{d}.mixer.in_proj.weight", .{i}, &.{ 64 + 96 + 2, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.conv1d.weight", .{i}, &.{ 96, 4, 1 }, .F32, 0.3);
                try t.rand("backbone.layers.{d}.mixer.conv1d.bias", .{i}, &.{96}, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.A_log", .{i}, &.{2}, .F32, 0.3);
                try t.fill("backbone.layers.{d}.mixer.D", .{i}, &.{2}, .F32, 1);
                try t.rand("backbone.layers.{d}.mixer.dt_bias", .{i}, &.{2}, .F32, 0.3);
                try t.fill("backbone.layers.{d}.mixer.norm.weight", .{i}, &.{64}, .F32, 1);
                try t.rand("backbone.layers.{d}.mixer.out_proj.weight", .{i}, &.{ h, 64 }, .F32, 0.1);
            },
            '*' => {
                try t.rand("backbone.layers.{d}.mixer.q_proj.weight", .{i}, &.{ 64, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.k_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.v_proj.weight", .{i}, &.{ 32, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.o_proj.weight", .{i}, &.{ h, 64 }, .F32, 0.1);
            },
            'E' => {
                try t.rand("backbone.layers.{d}.mixer.gate.weight", .{i}, &.{ 4, h }, .F32, 0.3);
                try t.rand("backbone.layers.{d}.mixer.gate.e_score_correction_bias", .{i}, &.{4}, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.switch_mlp.fc1.weight", .{i}, &.{ 4, 32, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.switch_mlp.fc2.weight", .{i}, &.{ 4, h, 32 }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.shared_experts.up_proj.weight", .{i}, &.{ 64, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.shared_experts.down_proj.weight", .{i}, &.{ h, 64 }, .F32, 0.1);
            },
            else => {
                try t.rand("backbone.layers.{d}.mixer.up_proj.weight", .{i}, &.{ 128, h }, .F32, 0.1);
                try t.rand("backbone.layers.{d}.mixer.down_proj.weight", .{i}, &.{ h, 128 }, .F32, 0.1);
            },
        }
    }
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes =
        \\{"model_type":"nemotron_h","vocab_size":64,"hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"max_position_embeddings":4096,
        \\ "layer_norm_epsilon":1e-5,"hybrid_override_pattern":"M*E-","mamba_num_heads":2,"mamba_head_dim":32,
        \\ "n_groups":1,"ssm_state_size":16,"conv_kernel":4,"n_routed_experts":4,"num_experts_per_tok":2,
        \\ "moe_intermediate_size":32,"moe_shared_expert_intermediate_size":64,"routed_scaling_factor":2.5,
        \\ "n_group":1,"topk_group":1,"norm_topk_prob":true,"eos_token_id":2}
        },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

// ── tests ──

test {
    _ = config;
}
