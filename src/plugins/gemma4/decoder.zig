//! The Gemma 4 dense decoder: the Gemma attention and GeGLU blocks between sandwich norms, per-layer embeddings (PLE)
//! added after each layer's MLP, a learned scalar on every layer's output, the logit softcap. Its own skeleton because
//! the PLE seam and the layer scalar fold into the residual kernels at decode, which `sdk/bases/decoder.zig`'s seams
//! do not carry. Serves prefill, decode, verify and batched rows.
//! copied_from: sdk/bases/decoder.zig (the layer loop), engines/mlx/forward/dense.zig (`forwardStandardWith`'s
//! seams and PLE order), engines/mlx/batched/dense.zig (the batched tick's), plugins/gemma4/ple.zig
//! (`computePLEInput`, `pleProject`), sdk/blocks/mlp/dense.zig (`residualTail`).

const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const residual_norm = @import("../../sdk/blocks/norms/residual_norm.zig");
const Attention = @import("../gemma3/load.zig").Attention;
const Compiled = @import("../gemma3/compiled.zig").Compiled;
const gelu_ple = @import("gelu_ple.zig");
const moe = @import("moe.zig");

const b = bases.blocks;
const Array = ops.Array;
const Graph = ops.Graph;
const Ctx = bases.Ctx;
const Linear = b.linear.Linear;
const Embed = b.embed.TableEmbed;
const Mlp = b.extras.GeGlu;
const Head = b.extras.SoftcapHead;

pub const Run = run_mod.Run;
pub const Out = run_mod.Out;

/// Every block serves batched rows.
pub const batches_decode = Attention.supports_batched_rows and Mlp.supports_batched_rows;

/// A layer's per-layer-embedding gate, projection and norm.
pub const Ple = struct { gate: Linear, proj: Linear, norm: Array };

/// A 26B-A4B layer's routed half: the second feed-forward input norm, the two branch norms and the experts.
pub const MoeTail = struct {
    pre_ff_norm_2: Array,
    post_ff_norm_1: Array,
    post_ff_norm_2: Array,
    experts: moe.Weights,
};

pub const Layer = struct {
    input_norm: Array,
    attn: Attention.Weights,
    post_attn_norm: Array,
    pre_ff_norm: Array,
    mlp: Mlp.Weights,
    post_ff_norm: Array,
    /// `layer_scalar` (one element); null = none.
    layer_scalar: Array = .{},
    ple: ?Ple = null,
    /// Routed experts beside `mlp` (then the shared expert, its output normed by `post_ff_norm` after the sum).
    moe: ?MoeTail = null,
};

pub const Weights = struct {
    embed: Embed.Weights,
    /// `embed_tokens_per_layer`, `per_layer_model_projection` and its norm; `ple_dim` 0 = no PLE.
    ple_table: Linear = .{},
    ple_proj: Linear = .{},
    ple_proj_norm: Array = .{},
    ple_dim: c_int = 0,
    layers: []const Layer,
    final_norm: Array,
    head: Head.Weights,
    compiled: *const Compiled,
};

/// `h + [rms(delta, w_pre)]`, times `scale`, then `rms(sum, w_post)` when asked: one dispatch at a solo decode step
/// where the kernel takes the shape, else the composed ops. `pre_post_eps`: `w_pre` norms at the post-norm epsilon.
pub fn seam(c: *Ctx, h: Array, delta: Array, w_pre: Array, pre_post_eps: bool, scale: Array, w_post: Array) !bases.block.End {
    const g = c.g;
    const sh = c.shared;
    if (c.mode == .decode) {
        if (try residual_norm.fusedResidualNorm(g.s, h, delta, .{ .w1 = w_pre, .eps1 = if (pre_post_eps) sh.post_eps_arr else sh.eps_arr, .scale = scale, .w_post = w_post, .eps = sh.eps_arr })) |r| {
            const sum = g.adopt(r.sum) catch |e| {
                if (r.normed.ctx != null) _ = mlx.mlx_array_free(r.normed);
                return e;
            };
            return .{ .h = sum, .normed = if (r.normed.ctx != null) try g.adopt(r.normed) else null };
        }
    }
    const nb = if (w_pre.ctx != null) try g.rmsNorm(delta, w_pre, if (pre_post_eps) sh.post_eps else sh.eps) else delta;
    var sum = try g.add(h, nb);
    if (scale.ctx != null) sum = try g.mul(sum, scale);
    return .{ .h = sum, .normed = if (w_post.ctx != null) try g.rmsNorm(sum, w_post, sh.eps) else null };
}

/// Every layer's per-layer input `[B, S, layers, ple_dim]`: the per-layer table rows of `ids` scaled by
/// `sqrt(ple_dim)`, plus the normed projection of the embeddings `h`, over `sqrt(2)`.
fn pleInput(w: *const Weights, c: *Ctx, ids: Array, h: Array) !Array {
    const g = c.g;
    const ish = ops.shape(ids);
    const bsz = ish[0];
    const seq = ish[1];
    const n_layers: c_int = @intCast(w.layers.len);
    const shape4 = [_]c_int{ bsz, seq, n_layers, w.ple_dim };
    const rows = try b.embed.gatherRows(g, &w.ple_table, try g.reshape(ids, &.{bsz * seq}), .bfloat16);
    const emb = try g.mul(try g.reshape(rows, &shape4), try g.scalar(@sqrt(@as(f32, @floatFromInt(w.ple_dim))), .bfloat16));
    const hidden: c_int = ops.shape(h)[2];
    const proj = try g.mul(try b.head.project(g, &w.ple_proj, h), try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hidden))), .bfloat16));
    const normed = try g.rmsNorm(try g.reshape(proj, &shape4), w.ple_proj_norm, c.shared.eps);
    return g.mul(try g.add(normed, emb), try g.scalar(1.0 / @sqrt(2.0), .bfloat16));
}

/// `per_layer_projection(gelu(gate(h)) * ple[layer])`, not yet normed or added. A one-row step is one fused dispatch
/// where it serves.
pub fn pleProject(compiled: *const Compiled, c: *Ctx, p: *const Ple, h: Array, ple_input: Array) !Array {
    const g = c.g;
    const sh = ops.shape(ple_input);
    const li: c_int = @intCast(c.layer);
    const slice = try g.slice(ple_input, &.{ 0, 0, li, 0 }, &.{ sh[0], sh[1], li + 1, sh[3] });
    const ple = try g.reshape(slice, &.{ sh[0], sh[1], sh[3] });
    const gate = try p.gate.forward(g, h);
    const dt = ops.dtype(gate);
    if (sh[0] * sh[1] == 1) blk: {
        const table = try compiled.geluTable(dt);
        if (table.ctx == null) break :blk;
        if (try gelu_ple.fusedGeluPleProj(g.s, gate, ple, table, p.proj.w, p.proj.s, p.proj.b, .{ .bits = p.proj.bits, .group_size = p.proj.group_size, .mode = p.proj.mode })) |y| return g.adopt(y);
    }
    const table: Array = if (fused_act.swigluFusedEnabled() and (dt == .bfloat16 or dt == .float16)) try compiled.geluTable(dt) else .{};
    const gated = if (table.ctx != null) (if (try fused_act.tableGateMul(g.s, .gelu_tanh, gate, ple, table, null)) |y| try g.adopt(y) else null) else null;
    return p.proj.forward(g, gated orelse try g.mul(try g.adopt(try compiled.geluOf(gate)), ple));
}

/// A 26B-A4B layer after attention (the engine's `gemma4MoeLayerTail`): the attention seam yields both feed-forward
/// inputs, the shared MLP reads one and the experts the other (routed off the raw residual), and the two outputs,
/// each normed, sum into one more norm before the residual add and the layer scalar. At a decode step each seam is
/// one dispatch.
fn moeTail(c: *Ctx, h: Array, attn: Array, L: *const Layer, mt: *const MoeTail, scalar: Array, next: Array) !bases.block.End {
    const g = c.g;
    const sh = c.shared;
    const decode = c.mode == .decode;
    var hm: Array = undefined;
    var shared_in: Array = undefined;
    var expert_in: Array = undefined;
    const seam1 = if (decode) try residual_norm.fusedResidualNorm(g.s, h, attn, .{ .w1 = L.post_attn_norm, .eps1 = sh.eps_arr, .w_post = L.pre_ff_norm, .w_post2 = mt.pre_ff_norm_2, .eps = sh.eps_arr }) else null;
    if (seam1) |r| {
        hm = try g.adopt(r.sum);
        shared_in = try g.adopt(r.normed);
        expert_in = try g.adopt(r.normed2);
    } else {
        hm = try g.add(h, try g.rmsNorm(attn, L.post_attn_norm, sh.eps));
        shared_in = try g.rmsNorm(hm, L.pre_ff_norm, sh.eps);
        expert_in = try g.rmsNorm(hm, mt.pre_ff_norm_2, sh.eps);
    }
    const shared_out = try Mlp.forward(&L.mlp, c, shared_in);
    const expert_out = try moe.forward(&mt.experts, g, sh.eps, hm, expert_in);
    if (decode) {
        if (try residual_norm.fusedResidualNorm(g.s, hm, shared_out, .{
            .b2 = expert_out,
            .w1 = mt.post_ff_norm_1,
            .eps1 = sh.eps_arr,
            .w2 = mt.post_ff_norm_2,
            .eps2 = sh.eps_arr,
            .wc = L.post_ff_norm,
            .epsc = sh.eps_arr,
            .scale = scalar,
            .w_post = next,
            .eps = sh.eps_arr,
        })) |r| {
            const sum = g.adopt(r.sum) catch |e| {
                if (r.normed.ctx != null) _ = mlx.mlx_array_free(r.normed);
                return e;
            };
            return .{ .h = sum, .normed = if (r.normed.ctx != null) try g.adopt(r.normed) else null };
        }
    }
    const combined = try g.add(try g.rmsNorm(shared_out, mt.post_ff_norm_1, sh.eps), try g.rmsNorm(expert_out, mt.post_ff_norm_2, sh.eps));
    var out = try g.add(hm, try g.rmsNorm(combined, L.post_ff_norm, sh.eps));
    if (scalar.ctx != null) out = try g.mul(out, scalar);
    return .{ .h = out, .normed = null };
}

/// A dense layer after attention: the sandwich-normed GeGLU, then the per-layer input's seam where the layer has one.
fn denseTail(w: *const Weights, c: *Ctx, h: Array, attn: Array, L: *const Layer, next: Array, ple_input: Array) !bases.block.End {
    const mid = try seam(c, h, attn, L.post_attn_norm, true, .{}, L.pre_ff_norm);
    const ff = mid.normed orelse try c.g.rmsNorm(mid.h, L.pre_ff_norm, c.shared.eps);
    const down = try Mlp.forward(&L.mlp, c, ff);
    const p = if (ple_input.ctx != null) L.ple else null;
    if (p == null) return seam(c, mid.h, down, L.post_ff_norm, true, L.layer_scalar, next);
    const end = try seam(c, mid.h, down, L.post_ff_norm, true, .{}, .{});
    return seam(c, end.h, try pleProject(w.compiled, c, &p.?, end.h, ple_input), p.?.norm, false, L.layer_scalar, next);
}

/// One forward of `ids` (`[slots, rows]` int32). Results are owned by `g`.
pub fn forward(w: *const Weights, shared: *const bases.Shared, g: *Graph, ids: Array, run: Run) !Out {
    var c = run.ctx(g, shared);
    // Masks built once for every layer (a sliding layer's band was rebuilt per layer).
    var memo: bases.block.Memo = .{ .g = g };
    c.memo = &memo;
    var h = try run.splice(g, try Embed.forward(&w.embed, &c, ids), ids);
    // Per-layer inputs read the text tokens only: an image position's id is 0 (the reference's `text_mask`).
    const img = run.placeholders.image;
    const ple_ids = if (run.media.ctx != null and img > 0) try g.where(try g.equal(ids, try g.scalar(@floatFromInt(img), .int32)), try g.scalar(0, .int32), ids) else ids;
    const ple_input: Array = if (w.ple_dim > 0) try pleInput(w, &c, ple_ids, h) else .{};

    var carried: ?Array = null;
    for (w.layers, 0..) |*L, i| {
        var lg = g.scope();
        defer lg.deinit();
        c.g = &lg;
        c.layer = @intCast(i);
        const fuse = c.mode == .decode;
        const next: Array = if (!fuse) .{} else if (i + 1 < w.layers.len) w.layers[i + 1].input_norm else w.final_norm;
        const normed = carried orelse try lg.rmsNorm(h, L.input_norm, shared.eps);
        const attn = try Attention.forward(&L.attn, &c, normed);
        const end = if (L.moe) |*mt|
            try moeTail(&c, h, attn, L, mt, L.layer_scalar, next)
        else
            try denseTail(w, &c, h, attn, L, next, ple_input);

        c.g = g;
        const old_h = h;
        h = try g.keep(end.h);
        g.drop(old_h);
        if (carried) |old| g.drop(old);
        carried = if (end.normed) |n| try g.keep(n) else null;
        try run.capture(g, @intCast(i), h);
        try run.cadence(i, w.layers.len, h);
    }
    c.g = g;
    const final = carried orelse try g.rmsNorm(h, w.final_norm, shared.eps);
    return run.finish(g, &c, final, Head, &w.head);
}
