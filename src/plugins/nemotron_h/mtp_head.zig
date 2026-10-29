//! Nemotron-H's MTP head behind `SpecHooks.head`: sevren-ai's `mtp_head.safetensors` (bare keys, a dense
//! `eh_proj`) or TensorFold's `mtp-4bit.safetensors` (`layers.0/1`). x = eh_proj([enorm(emb(ids)), hnorm(h)]);
//! x += attn(attn.norm(x)) (NoPE, no q/k norm, no output gate); x += moe(moe.norm(x)) (the trunk's MoE, unfolded);
//! the output is final_norm(x), which the engine projects through the trunk head. `h` is the trunk's final-normed
//! hidden. The head's own KV cache is layer 0 of `caches[0]`.
//! copied_from: sdk/spec/mtp/mtp.zig (`loadNemotronMtp`, `fcConcat`, `frontChain`, `backChain`, `appendKvOnly`,
//! `forwardWithMrope` without M-RoPE, `qLinearFwd`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const kv = @import("../../sdk/bases/kv.zig");
const reader = @import("../../sdk/bases/reader.zig");
const model = @import("../../sdk/weights/load.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const NemotronMoe = @import("../../sdk/bases/blocks/nemotron_moe.zig").NemotronMoe;
const verifyQmm = @import("../../sdk/blocks/linear/verify_qmm.zig").verifyQmm;
const attn256 = @import("../../sdk/blocks/attention/attn256.zig");
const sdpa = @import("../../sdk/blocks/attention/sdpa.zig");
const config = @import("config.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

/// The head's tensor names: sevren-ai's bare keys or TensorFold's module names.
const Names = struct {
    attn: []const u8,
    moe: []const u8,
    eh_proj: []const u8,
    enorm: []const u8,
    hnorm: []const u8,
    final_norm: []const u8,

    const sevren = Names{ .attn = "attn", .moe = "moe", .eh_proj = "eh_proj", .enorm = "enorm", .hnorm = "hnorm", .final_norm = "final_norm" };
    const tensorfold = Names{ .attn = "layers.0", .moe = "layers.1", .eh_proj = "layers.0.eh_proj", .enorm = "layers.0.enorm.weight", .hnorm = "layers.0.hnorm.weight", .final_norm = "layers.1.final_layernorm.weight" };
};

/// The sidecar files a Nemotron-H head ships as, in the engine's order.
const files = [_][]const u8{ "mtp_head.safetensors", "mtp-4bit.safetensors" };

pub const Head = struct {
    owned: reader.Owned = .{},
    eh_proj: Linear,
    enorm: Array,
    hnorm: Array,
    final_norm: Array,
    attn_norm: Array,
    moe_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    moe: NemotronMoe.Weights,
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    shared: bases.Shared,

    pub fn deinit(h: *Head) void {
        h.shared.deinit();
        h.owned.deinit();
    }
};

/// The head in `dir`, read through the trunk's quant block; null when the pack ships none.
pub fn load(ctx: *const abi.LoadCtx, c: *const config.Config) !?*Head {
    const rel = for (files) |f| {
        if (ctx.dir.has(f)) break f;
    } else return null;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ ctx.model_dir.slice(), rel });
    var res = Resident.adopt(try model.loadWeightsSingleFile(gpa, path));
    defer res.deinit();
    const src = res.source();
    var lctx = ctx.*;
    lctx.weights = &src;
    var r = try reader.Reader.open(&lctx, c.block_json, "");
    defer r.close();
    const names = if (r.has("attn.mixer.q_proj.weight"))
        Names.sevren
    else if (r.has("layers.0.eh_proj.weight") and r.has("layers.1.final_layernorm.weight"))
        Names.tensorfold
    else
        return error.UnknownMtpLayout;

    const h = try gpa.create(Head);
    errdefer gpa.destroy(h);
    var own: reader.Owned = .{};
    errdefer own.deinit();
    var nb: [160]u8 = undefined;
    const hidden = c.hidden_size;
    // A bare dense `eh_proj` `[H, 2H]` is read pre-transposed; a `.weight` triple goes through the format.
    const eh_proj: Linear = if (r.has(names.eh_proj)) blk: {
        const raw = try r.need(names.eh_proj);
        defer _ = mlx.mlx_array_free(raw);
        var t = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_transpose_axes(&t, raw, &[_]c_int{ 1, 0 }, 2, r.s));
        break :blk .{ .w = t };
    } else try r.proj(names.eh_proj, 2 * hidden);
    try own.linear(eh_proj);
    var norms: [5]Array = undefined;
    for (&norms, [_][]const u8{ names.enorm, names.hnorm, names.final_norm, try std.fmt.bufPrint(&nb, "{s}.norm.weight", .{names.attn}), "" }, 0..) |*n, key, i| {
        var kb: [160]u8 = undefined;
        n.* = try r.need(if (i == 4) try std.fmt.bufPrint(&kb, "{s}.norm.weight", .{names.moe}) else key);
        try own.own(n.*);
    }
    var p: [4]Linear = undefined;
    for (&p, [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj" }, 0..) |*l, leaf, k| {
        l.* = try r.proj(try std.fmt.bufPrint(&nb, "{s}.mixer.{s}", .{ names.attn, leaf }), if (k == 3) c.num_attention_heads * c.head_dim else hidden);
        try own.linear(l.*);
    }
    const moe = try @import("plugin.zig").moeOf(&own, &r, c, names.moe);
    for (own.list.items) |a| _ = mlx.mlx_array_eval(a);
    h.* = .{
        .owned = own,
        .eh_proj = eh_proj,
        .enorm = norms[0],
        .hnorm = norms[1],
        .final_norm = norms[2],
        .attn_norm = norms[3],
        .moe_norm = norms[4],
        .q = p[0],
        .k = p[1],
        .v = p[2],
        .o = p[3],
        .moe = moe,
        .heads = @intCast(c.num_attention_heads),
        .kv_heads = @intCast(c.num_key_value_heads),
        .head_dim = @intCast(c.head_dim),
        .shared = bases.Shared.init(c.rms_norm_eps, c.rms_norm_eps),
    };
    return h;
}

/// A head projection as the legacy head reads it: dense plain matmul; affine through the verify-width kernel where
/// it serves, else MLX's quantized matmul.
pub fn lin(g: *Graph, l: *const Linear, x: Array) !Array {
    if (l.s.ctx == null) return g.matmul(x, l.w);
    if (l.mode == .affine) if (try verifyQmm(g.s, x, l.w, l.s, l.b, l.bits, l.group_size)) |y| return g.adopt(y);
    return g.call(mlx.mlx_quantized_matmul, .{ x, l.w, l.s, l.b, true, mlx.mlx_optional_int.some(@intCast(l.group_size)), mlx.mlx_optional_int.some(@intCast(l.bits)), l.mode.cstr() });
}

/// eh_proj over [enorm(emb(ids)), hnorm(hidden)], both in the head's dtype (`hnorm`'s).
pub fn front(g: *Graph, table: *const Linear, ids: Array, hidden: Array, enorm: Array, hnorm: Array, eh_proj: *const Linear, eps: f32) !Array {
    const n = ops.shape(ids)[0];
    const dt = ops.dtype(hnorm);
    const emb = try g.reshape(try bases.blocks.embed.gatherRows(g, table, ids, .bfloat16), &.{ 1, n, ops.shape(hidden)[2] });
    const e = try g.rmsNorm(try castTo(g, emb, dt), enorm, eps);
    const hh = try g.rmsNorm(try castTo(g, hidden, dt), hnorm, eps);
    return lin(g, eh_proj, try g.concat(&.{ e, hh }, 2));
}

fn castTo(g: *Graph, x: Array, dt: mlx.mlx_dtype) !Array {
    return if (ops.dtype(x) == dt) x else g.astype(x, dt);
}

/// k and v `[1, kv_heads, S, D]` of the normed rows, appended to the head's cache.
fn appendKv(h: *const Head, c: *bases.Ctx, normed: Array, s: c_int) !abi.KvView {
    const g = c.g;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const k = try g.transpose(try g.reshape(try lin(g, &h.k, normed), &.{ 1, s, h.kv_heads, h.head_dim }), &perm);
    const v = try g.transpose(try g.reshape(try lin(g, &h.v, normed), &.{ 1, s, h.kv_heads, h.head_dim }), &perm);
    return kv.append(c, k, v, 0);
}

/// History rows: K/V only.
pub fn kvOnly(h: *const Head, c: *bases.Ctx, table: *const Linear, ids: Array, hidden: Array) !void {
    const x = try front(c.g, table, ids, hidden, h.enorm, h.hnorm, &h.eh_proj, h.shared.eps);
    var view = try appendKv(h, c, try c.g.rmsNorm(x, h.attn_norm, h.shared.eps), ops.shape(ids)[0]);
    kv.release(c, &view);
}

/// The full layer over `ids`; the last row's output `[1, 1, hidden]`.
pub fn full(h: *const Head, c: *bases.Ctx, table: *const Linear, ids: Array, hidden: Array) !Array {
    const g = c.g;
    const s = ops.shape(ids)[0];
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const x = try front(g, table, ids, hidden, h.enorm, h.hnorm, &h.eh_proj, h.shared.eps);
    const normed = try g.rmsNorm(x, h.attn_norm, h.shared.eps);
    const q = try g.transpose(try g.reshape(try lin(g, &h.q, normed), &.{ 1, s, h.heads, h.head_dim }), &perm);
    var view = try appendKv(h, c, normed, s);
    defer kv.release(c, &view);
    const dv = kv.dense(view);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(h.head_dim)));
    const attn = (if (s > 1) try bases.blocks.decoder_attention.own(g, try attn256.fusedSdpa256Prefill(g.s, q, dv.k, dv.v, scale, 0)) else null) orelse
        try g.call(mlx.mlx_fast_scaled_dot_product_attention, .{ q, dv.k, dv.v, scale, if (s > 1) "causal" else "", Array{}, Array{}, s > 1 and sdpa.sdpaForceFused(q, dv.k) });
    const o = try lin(g, &h.o, try g.reshape(try g.transpose(attn, &perm), &.{ 1, s, h.heads * h.head_dim }));
    const h1 = try g.add(x, o);
    const m = try NemotronMoe.forward(&h.moe, c, try g.rmsNorm(h1, h.moe_norm, h.shared.eps));
    const y = try g.rmsNorm(try g.add(h1, m), h.final_norm, h.shared.eps);
    if (s == 1) return y;
    return g.slice(y, &.{ 0, s - 1, 0 }, &.{ 1, s, ops.shape(y)[2] });
}

/// `HeadHooks.forward`: `.last_row` appends the leading rows' K/V and runs the last; `.none` with `want_hidden` runs
/// every row and reads the last; `.none` alone appends history.
pub fn forward(h: *const Head, table: *const Linear, s: ops.Stream, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    return step(@This(), h, table, s, f, out);
}

/// One head step over module `M`'s `kvOnly` (history rows) and `full` (the layer, the last row's output) on its
/// `Head`, as the legacy `appendKvOnly` / `forwardWithMrope` split it: the last row of a `.last_row` step runs at its
/// own position.
pub fn step(comptime M: type, h: *const M.Head, table: *const Linear, s: ops.Stream, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    var g = Graph.init(gpa, s);
    defer g.deinit();
    const rows: c_int = @intCast(f.rows);
    const ids = if (f.ids.ctx != null) try g.reshape(.{ .ctx = f.ids.ctx }, &.{rows}) else blk: {
        const host = try gpa.alloc(i32, f.rows);
        defer gpa.free(host);
        for (f.tokens.slice(), host) |t, *o| o.* = @intCast(t);
        break :blk try g.fromSlice(i32, host, &.{rows});
    };
    const hidden: Array = .{ .ctx = f.hidden_in.ctx };
    var c: bases.Ctx = .{ .g = &g, .mode = if (rows == 1) .decode else .prefill, .rows = rows, .positions = f.positions.slice(), .caches = f.caches.slice(), .cache_ops = f.cache_ops, .shared = &h.shared };
    if (f.logits == .none and !f.want_hidden) return M.kvOnly(h, &c, table, ids, hidden);
    var last_pos: [1]u32 = undefined;
    const y = if (f.logits == .last_row and rows > 1) blk: {
        const hs = ops.shape(hidden);
        try M.kvOnly(h, &c, table, try g.slice(ids, &.{0}, &.{rows - 1}), try g.slice(hidden, &.{ 0, 0, 0 }, &.{ 1, rows - 1, hs[2] }));
        last_pos = .{c.offset() + @as(u32, @intCast(rows - 1))};
        c.positions = &last_pos;
        c.rows = 1;
        c.mode = .decode;
        break :blk try M.full(h, &c, table, try g.slice(ids, &.{rows - 1}, &.{rows}), try g.slice(hidden, &.{ 0, rows - 1, 0 }, &.{ 1, rows, hs[2] }));
    } else try M.full(h, &c, table, ids, hidden);
    out.hidden = .{ .ctx = g.escape(y).ctx };
}
