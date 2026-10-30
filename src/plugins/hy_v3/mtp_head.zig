//! Hunyuan 3's MTP head behind `SpecHooks.head`: `model-mtp.safetensors` (or the trunk's shards), `mtp.*` keys.
//! x = eh_proj([enorm(emb(ids)), hnorm(h)]); x += attn(input_norm(x)) (a full hy3 layer: per-head q/k RMS norms,
//! RoPE, no output gate); x += moe(post_norm(x)) (sigmoid routing with a selection-only bias, an ungated shared
//! expert); the output is final_layernorm(x), which the engine projects through the trunk head. `h` is the trunk's
//! final-normed hidden. The head's own KV cache is layer 0 of `caches[0]`.
//! copied_from: sdk/spec/mtp/mtp.zig (`loadHy3Mtp`, `frontChain`, `backChain`, `appendKvOnly`, `forwardWithMrope`
//! without M-RoPE).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const kv = @import("../../sdk/bases/kv.zig");
const moe = @import("../../sdk/bases/blocks/moe.zig");
const compiled = @import("../../sdk/bases/blocks/compiled.zig");
const mtp_mod = @import("../../sdk/spec/mtp/source.zig");
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const sdpa = @import("../../sdk/blocks/attention/sdpa.zig");
const reader = @import("../../sdk/bases/reader.zig");
const nemotron_head = @import("../nemotron_h/mtp_head.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const config = @import("config.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const lin = nemotron_head.lin;
const gpa = std.heap.c_allocator;

pub const Head = struct {
    owned: reader.Owned = .{},
    eh_proj: Linear,
    enorm: Array,
    hnorm: Array,
    final_norm: Array,
    input_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    moe: moe.Moe.Weights,
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    rope_dims: c_int,
    rope_base: f32,
    scale: f32,
    shared: bases.Shared,

    pub fn deinit(h: *Head) void {
        compiled.free(&h.moe.route_fn);
        h.shared.deinit();
        h.owned.deinit();
    }
};

/// The head the pack ships, read through the trunk's quant block; null when it ships none.
pub fn load(ctx: *const abi.LoadCtx, c: *const config.Config) !?*Head {
    var res = Resident.adopt(try reader.headWeights(ctx, &mtp_mod.sidecar_rel_paths, "mtp.eh_proj.weight", "mtp."));
    defer res.deinit();
    const src = res.source();
    var lctx = ctx.*;
    lctx.weights = &src;
    var r = try reader.Reader.open(&lctx, c.block_json, "mtp");
    defer r.close();
    if (!r.has("mtp.eh_proj.weight")) return null;

    var own: reader.Owned = .{};
    errdefer own.deinit();
    var nb: [160]u8 = undefined;
    const hidden = c.hidden_size;
    var norms: [7]Array = undefined;
    for (&norms, [_][]const u8{ "enorm", "hnorm", "final_layernorm", "layer.input_layernorm", "layer.post_attention_layernorm", "layer.self_attn.q_norm", "layer.self_attn.k_norm" }) |*n, leaf| {
        n.* = try r.need(try r.name(&nb, "{s}.weight", .{leaf}));
        try own.own(n.*);
    }
    var p: [5]Linear = undefined;
    for (&p, [_][]const u8{ "eh_proj", "layer.self_attn.q_proj", "layer.self_attn.k_proj", "layer.self_attn.v_proj", "layer.self_attn.o_proj" }, [_]u32{ 2 * hidden, hidden, hidden, hidden, c.num_attention_heads * c.head_dim }) |*l, leaf, in_dim| {
        l.* = try r.proj(try r.name(&nb, "{s}", .{leaf}), in_dim);
        try own.linear(l.*);
    }
    const router = try r.proj(try r.name(&nb, "layer.mlp.router.gate", .{}), hidden);
    try own.linear(router);
    var banks: [3]moe.Bank = undefined;
    for (&banks, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, c.moe_intermediate }) |*bk, leaf, in_dim| {
        const l = try r.proj(try r.name(&nb, "layer.mlp.experts.{s}", .{leaf}), in_dim);
        try own.linear(l);
        // The engine's head binds a dense bank untransposed and `gather_mm` refuses it: nothing to be equal to.
        if (l.s.ctx == null) return error.DenseExpertHead;
        bk.* = reader.bank(l, c.quant);
    }
    const bias = try r.need(try r.name(&nb, "layer.mlp.expert_bias", .{}));
    try own.own(bias);
    var sh: [3]Linear = undefined;
    for (&sh, [_][]const u8{ "gate_proj", "up_proj", "down_proj" }, [_]u32{ hidden, hidden, c.shared_intermediate }) |*l, leaf, in_dim| {
        l.* = try r.proj(try r.name(&nb, "layer.mlp.shared_mlp.{s}", .{leaf}), in_dim);
        try own.linear(l.*);
    }
    for (own.list.items) |a| _ = mlx.mlx_array_eval(a);

    const h = try gpa.create(Head);
    errdefer gpa.destroy(h);
    h.* = .{
        .owned = own,
        .eh_proj = p[0],
        .enorm = norms[0],
        .hnorm = norms[1],
        .final_norm = norms[2],
        .input_norm = norms[3],
        .post_norm = norms[4],
        .q_norm = norms[5],
        .k_norm = norms[6],
        .q = p[1],
        .k = p[2],
        .v = p[3],
        .o = p[4],
        .moe = .{
            .router = router,
            .route = .{ .sigmoid_bias = .{ .expert_bias = bias, .norm = c.route_norm, .scale = c.route_scale } },
            .gate = banks[0],
            .up = banks[1],
            .down = banks[2],
            .shared = .{ .gate = sh[0], .up = sh[1], .down = sh[2] },
            .top_k = @intCast(c.top_k),
            .experts = @intCast(c.experts),
        },
        .heads = @intCast(c.num_attention_heads),
        .kv_heads = @intCast(c.num_key_value_heads),
        .head_dim = @intCast(c.head_dim),
        .rope_dims = @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor),
        .rope_base = c.rope_theta,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.query_pre_attn_scalar))),
        .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps),
    };
    // The trunk's routing chain, compiled as the trunk compiles it.
    h.moe.route_fn = moe.compileRouting(&h.moe, r.s);
    return h;
}

fn front(h: *const Head, g: *Graph, table: *const Linear, ids: Array, hidden: Array) !Array {
    return nemotron_head.front(g, table, ids, hidden, h.enorm, h.hnorm, &h.eh_proj, h.shared.eps);
}

/// Roped k and v `[1, kv_heads, S, D]` of the normed rows, appended to the head's cache.
fn appendKv(h: *const Head, c: *bases.Ctx, normed: Array, s: c_int) !abi.KvView {
    const g = c.g;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const k = try g.transpose(try g.rmsNorm(try g.reshape(try lin(g, &h.k, normed), &.{ 1, s, h.kv_heads, h.head_dim }), h.k_norm, h.shared.eps), &perm);
    const v = try g.transpose(try g.reshape(try lin(g, &h.v, normed), &.{ 1, s, h.kv_heads, h.head_dim }), &perm);
    return kv.append(c, try g.rope(k, h.rope_dims, h.rope_base, @intCast(c.offset())), v, 0);
}

/// History rows: K/V only.
pub fn kvOnly(h: *const Head, c: *bases.Ctx, table: *const Linear, ids: Array, hidden: Array) !void {
    const x = try front(h, c.g, table, ids, hidden);
    var view = try appendKv(h, c, try c.g.rmsNorm(x, h.input_norm, h.shared.eps), ops.shape(ids)[0]);
    kv.release(c, &view);
}

/// The full layer over `ids`; the last row's output `[1, 1, hidden]`.
pub fn full(h: *const Head, c: *bases.Ctx, table: *const Linear, ids: Array, hidden: Array) !Array {
    const g = c.g;
    const s = ops.shape(ids)[0];
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const x = try front(h, g, table, ids, hidden);
    const normed = try g.rmsNorm(x, h.input_norm, h.shared.eps);
    const q_heads = try g.rmsNorm(try g.reshape(try lin(g, &h.q, normed), &.{ 1, s, h.heads, h.head_dim }), h.q_norm, h.shared.eps);
    const q = try g.rope(try g.transpose(q_heads, &perm), h.rope_dims, h.rope_base, @intCast(c.offset()));
    var view = try appendKv(h, c, normed, s);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const attn = (if (s > 1) try bases.blocks.decoder_attention.own(g, try attn256.fusedSdpa256Prefill(g.s, q, dv.k, dv.v, h.scale, 0)) else null) orelse
        try g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, dv.k, dv.v, h.scale, if (s > 1) "causal" else "", Array{}, Array{}, s > 1 and sdpa.sdpaForceFused(q, dv.k) });
    const o = try lin(g, &h.o, try g.reshape(try g.transpose(attn, &perm), &.{ 1, s, h.heads * h.head_dim }));
    const h1 = try g.add(x, o);
    const m = try moe.Moe.forward(&h.moe, c, try g.rmsNorm(h1, h.post_norm, h.shared.eps));
    const y = try g.rmsNorm(try g.add(h1, m), h.final_norm, h.shared.eps);
    if (s == 1) return y;
    return g.slice(y, &.{ 0, s - 1, 0 }, &.{ 1, s, ops.shape(y)[2] });
}

/// `HeadHooks.forward`, the legacy head's step split (`nemotron_h/mtp_head.zig` `step`).
pub fn forward(h: *const Head, table: *const Linear, s: ops.Stream, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    return nemotron_head.step(@This(), h, table, s, f, out);
}
