//! Hy4-flash's DFly2 drafter (`HYV4FlashDFly`): 5 dense MLA layers over the target's KV history, fused with
//! the last accepted token's hidden, then a serial hidden-correction AR head that samples the block's tokens
//! through the trunk's LM head. HF transformers does not implement it; semantics: SEMANTICS.md §8 / SGLang
//! #51159 / vLLM #68296.
//!
//! A round runs the drafter on `[t1, mask, mask, ...]`, verifies the block in ONE target forward (appended to
//! the host cache), keeps the drafts whose target argmax agrees, and truncates the rejected rows. Draft layers
//! read target layer 25+4n's KV view (the block's own rows are local); the indexer snapshot is not used (full
//! causal history — the target's verification keeps the output distribution exact).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const abi = @import("../../../abi/plugin.zig");
const proj = @import("proj.zig");
const config = @import("../config.zig");
const weights_load = @import("../../../sdk/weights/load.zig");
const log = @import("../../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Proj = proj.Proj;
const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const C = config.Config;

pub const TargetLayers = [_]u32{ 25, 29, 33, 37, 41 };

pub const Layer = struct {
    input_norm: Array,
    post_attn_norm: Array,
    q_a: Proj,
    q_a_norm: Array,
    q_b: Proj,
    kv_a: Proj,
    kv_a_norm: Array,
    wk: Array, // [heads, nope, rank]
    wv: Array, // [heads, v_dim, rank]
    linear_gate: Proj,
    o_proj: Proj,
    post_proj_norm: Array,
    sink: Array,
    gate: Proj,
    up: Proj,
    down: Proj,
    heads: u32,
    nope: u32,
    rope_dim: u32,
    v_dim: u32,
    rank: u32,
    theta: f32,
};

pub const Drafter = struct {
    layers: []Layer,
    final_norm: Array,
    enorm: Array,
    hnorm: Array,
    eh_proj: Proj,
    fusion_gate: Array,
    hc_hidden_norm: Array,
    hc_embed_norm: Array,
    hc_gate: Proj,
    hc_up: Proj,
    hc_down: Proj,
    owned: std.ArrayList(Array),
    c: *const C,
    hidden: u32,
    mask_id: u32,
    rms_eps: f32,
    s: mlx.mlx_stream,

    pub fn deinit(self: *Drafter) void {
        gpa.free(self.layers);
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(gpa);
        gpa.destroy(self);
    }
};

pub fn load(c: *const C, w: *const weights_load.Weights) !*Drafter {
    if (c.dfly_layers == 0) return error.NoDrafter;
    var owned: std.ArrayList(Array) = .empty;
    errdefer {
        for (owned.items) |x| _ = mlx.mlx_array_free(x);
        owned.deinit(gpa);
    }
    const d = try gpa.create(Drafter);
    errdefer gpa.destroy(d);
    const layers = try gpa.alloc(Layer, c.dfly_layers);
    errdefer gpa.free(layers);
    for (layers, 0..) |*l, i| {
        const n: u32 = @intCast(i);
        const pfx = "model.dfly_layers.{d}";
        const kv_w = try require(w, pfx ++ ".self_attn.kv_b_proj.weight", .{n});
        const halves = try denseHalves(&owned, kv_w, c.num_attention_heads, c.qk_nope_head_dim, c.v_head_dim, c.kv_lora_rank);
        l.* = .{
            .input_norm = try require(w, pfx ++ ".input_layernorm.weight", .{n}),
            .post_attn_norm = try require(w, pfx ++ ".post_attention_layernorm.weight", .{n}),
            .q_a = try denseBank(&owned, w, pfx ++ ".self_attn.q_a_proj", .{n}),
            .q_a_norm = try require(w, pfx ++ ".self_attn.q_a_layernorm.weight", .{n}),
            .q_b = try denseBank(&owned, w, pfx ++ ".self_attn.q_b_proj", .{n}),
            .kv_a = try denseBank(&owned, w, pfx ++ ".self_attn.kv_a_proj_with_mqa", .{n}),
            .kv_a_norm = try require(w, pfx ++ ".self_attn.kv_a_layernorm.weight", .{n}),
            .wk = halves[0],
            .wv = halves[1],
            .linear_gate = try denseBank(&owned, w, pfx ++ ".self_attn.linear_gate", .{n}),
            .o_proj = try denseBank(&owned, w, pfx ++ ".self_attn.o_proj", .{n}),
            .post_proj_norm = try require(w, pfx ++ ".self_attn.post_proj_norm.weight", .{n}),
            .sink = try require(w, pfx ++ ".self_attn.learnable_sink_param", .{n}),
            .gate = try denseBank(&owned, w, pfx ++ ".mlp.gate_proj", .{n}),
            .up = try denseBank(&owned, w, pfx ++ ".mlp.up_proj", .{n}),
            .down = try denseBank(&owned, w, pfx ++ ".mlp.down_proj", .{n}),
            .heads = c.num_attention_heads,
            .nope = c.qk_nope_head_dim,
            .rope_dim = c.qk_rope_head_dim,
            .v_dim = c.v_head_dim,
            .rank = c.kv_lora_rank,
            .theta = c.rope_theta_dsa,
        };
    }
    d.* = .{
        .layers = layers,
        .final_norm = try require(w, "model.dfly_layers.final_norm.weight", .{}),
        .enorm = try require(w, "model.dfly_layers.mtp_fusion.enorm.weight", .{}),
        .hnorm = try require(w, "model.dfly_layers.mtp_fusion.hnorm.weight", .{}),
        .eh_proj = try denseBank(&owned, w, "model.dfly_layers.mtp_fusion.eh_proj", .{}),
        .fusion_gate = try require(w, "model.dfly_layers.mtp_fusion.gate", .{}),
        .hc_hidden_norm = try require(w, "model.dfly_layers.hidden_correction.hidden_norm.weight", .{}),
        .hc_embed_norm = try require(w, "model.dfly_layers.hidden_correction.embed_norm.weight", .{}),
        .hc_gate = try denseBank(&owned, w, "model.dfly_layers.hidden_correction.gate_proj", .{}),
        .hc_up = try denseBank(&owned, w, "model.dfly_layers.hidden_correction.up_proj", .{}),
        .hc_down = try denseBank(&owned, w, "model.dfly_layers.hidden_correction.down_proj", .{}),
        .owned = owned,
        .c = c,
        .hidden = c.hidden_size,
        .mask_id = c.dfly_mask_token_id,
        .rms_eps = c.rms_norm_eps,
        .s = mlx.gpuStream(),
    };
    log.info("DFly2 drafter: {d} draft layers over target KV layers {any}, block <= {d}\n", .{ c.dfly_layers, TargetLayers[0..c.dfly_target_layer_count], @as(u32, 6) });
    return d;
}



fn require(w: *const weights_load.Weights, comptime fmt: []const u8, args: anytype) !Array {
    const n = try std.fmt.allocPrint(gpa, fmt, args);
    defer gpa.free(n);
    return w.get(n) orelse error.MissingWeight;
}

fn denseBank(owned: *std.ArrayList(Array), w: *const weights_load.Weights, comptime fmt: []const u8, args: anytype) !Proj {
    const x = try require(w, fmt ++ ".weight", args);
    var t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, x, &[_]c_int{ 1, 0 }, 2, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(t));
    try owned.append(gpa, t);
    return .{ .w = t };
}

fn denseHalves(owned: *std.ArrayList(Array), weight: Array, heads: u32, nope: u32, vd: u32, rank: u32) ![2]Array {
    const h: c_int = @intCast(heads);
    const np: c_int = @intCast(nope);
    const v: c_int = @intCast(vd);
    const r: c_int = @intCast(rank);
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, weight, &[_]c_int{ h, np + v, r }, 3, mlx.gpuStream()));
    var out: [2]Array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out[0]);
        _ = mlx.mlx_array_free(out[1]);
    }
    try mlx.check(mlx.mlx_slice(&out[0], shaped, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ h, np, r }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()));
    try mlx.check(mlx.mlx_slice(&out[1], shaped, &[_]c_int{ 0, np, 0 }, 3, &[_]c_int{ h, np + v, r }, 3, &[_]c_int{ 1, 1, 1 }, 3, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(out[0]));
    try mlx.check(mlx.mlx_array_eval(out[1]));
    try owned.append(gpa, out[0]);
    try owned.append(gpa, out[1]);
    return out;
}

