//! Inkling 2's trunk (mlx-lm `inkling2/model.py` `LanguageModel`): the normed embedding, the visit schedule (prelude,
//! the looped core run `passes` times, tail), the attention-residual block states every layer reads its inputs from,
//! the looped passes' messages, one conv-state entry per visit, the final depth read, the norm and the softcapped
//! head. Visit `v` owns KV slot and recurrent entry `v`.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const block = @import("../../sdk/bases/block.zig");
const kv = @import("../../sdk/bases/kv.zig");
const run_mod = @import("../../sdk/bases/run.zig");
const embed = @import("../../sdk/bases/blocks/embed.zig");
const head = @import("../../sdk/bases/blocks/head.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const attention = @import("attention.zig");
const conv = @import("conv.zig");
const depth = @import("depth.zig");
const moe = @import("moe.zig");

const Array = ops.Array;
const Graph = ops.Graph;

/// The base `run.tableForward` drives: this file (`Weights`, `batches_decode`, `forward`).
pub const Trunk = @This();

pub const Layer = struct {
    global: bool,
    boundary: bool,
    attn: attention.Weights,
    /// Sliding: `input_layernorm`; global: `q_input_layernorm`.
    norm_q: Array,
    /// Global: `k_input_layernorm` / `v_input_layernorm`; null ctx on sliding layers.
    norm_k: Array = .{},
    norm_v: Array = .{},
    /// f32 `[Q, D]` (Q = 1 sliding, 3 global); null ctx on layer 0.
    attn_query: Array = .{},
    /// f32 `[1, D]`.
    mlp_query: Array,
    post_attn_norm: Array,
    /// `[D, K, 1]`.
    post_attn_conv: Array,
    post_mlp_conv: Array,
    moe: moe.Weights,
    /// q, k, v, mlp; core layers only.
    gates: ?[4]depth.Gate = null,
};

pub const Weights = struct {
    /// Checkpoint layout `[V, D]`.
    embed: Linear,
    embed_norm: Array,
    layers: []Layer,
    /// f32 `[1, D]`.
    final_query: Array,
    norm: Array,
    /// `lm_head` with its rows cut to the unpadded vocab.
    head: Linear,
    /// f32 `[core_len, 4, D]`; null ctx without a loop.
    message_query: Array = .{},
    /// Visit -> physical layer.
    visits: []const u8,
    core_start: u32,
    core_len: u32,
    passes: u32,
    n_prelude_blocks: u32,
    hidden: u32,
    eps: f32,
    softcap: f32,
    /// f32 `[D]` of ones.
    ones_h: Array,
};

pub const batches_decode = false;

/// The weighted norm over an f32 input (the product rounds once, in f32); f32 out.
fn rms(g: *Graph, x: Array, w: Array, eps: f32) !Array {
    return g.rmsNorm(try g.astype(x, .float32), w, eps);
}

/// The unweighted norm of an f32 input.
fn urms(g: *Graph, x: Array, ones: Array) !Array {
    return g.rmsNorm(x, ones, 1e-6);
}

/// The head over the final rows: f32 logits softcapped (the trim to the unpadded vocab is the head's rows).
const Head = struct {
    pub const Weights = Trunk.Weights;

    pub fn forward(w: *const Trunk.Weights, c: *block.Ctx, x: Array) !Array {
        const g = c.g;
        const cap = try g.scalar(w.softcap, .float32);
        const lg = try g.astype(try head.project(g, &w.head, x), .float32);
        return g.mul(try g.tanh(try g.div(lg, cap)), cap);
    }
};

/// The three previous k/v/attn/mlp windows of a visit's conv state (null before its first write).
const Prev = struct { k: ?Array = null, v: ?Array = null, attn: ?Array = null, mlp: ?Array = null };

fn prevWindows(g: *Graph, st: ?Array, kv_width: c_int, d: c_int) !Prev {
    const s = st orelse return .{};
    return .{
        .k = try g.sliceAxis(s, 2, 0, kv_width),
        .v = try g.sliceAxis(s, 2, kv_width, 2 * kv_width),
        .attn = try g.sliceAxis(s, 2, 2 * kv_width, 2 * kv_width + d),
        .mlp = try g.sliceAxis(s, 2, 2 * kv_width + d, 2 * kv_width + 2 * d),
    };
}

/// One visit of layer `lay` on the block residual `br`: returns the next hidden (dtype `dt`) and the block states
/// after it (`bs` itself unless the layer opens a block). `msgs`: f32 `[1, L, 4, D]` on a looped pass, else null.
fn visit(w: *const Weights, lay: *const Layer, c: *block.Ctx, br: Array, bs: ?Array, msgs: ?Array, dt: ops.Dtype) !struct { h: Array, bs: ?Array } {
    const g = c.g;
    const eps = w.eps;
    const d: c_int = @intCast(w.hidden);
    const kv_width: c_int = @intCast(lay.attn.kv_heads * lay.attn.head_dim);
    const prev = try prevWindows(g, try kv.state(c, .conv), kv_width, d);

    const nr = try urms(g, try g.astype(br, .float32), w.ones_h);
    const ai: ?Array = if (lay.attn_query.ctx != null) try depth.attend(g, bs, nr, lay.attn_query) else null;
    var xs: [3]Array = undefined;
    if (lay.global) {
        for (&xs, 0..) |*x, j| x.* = if (ai) |a| try depth.row(g, a, @intCast(j)) else br;
    } else {
        const x0 = if (ai) |a| try depth.row(g, a, 0) else br;
        xs = .{ x0, x0, x0 };
    }
    if (msgs) |m| {
        const gates = lay.gates orelse return error.MissingMessageGates;
        for (&xs, 0..) |*x, j| x.* = try depth.gate(g, &gates[j], x.*, try depth.row(g, m, @intCast(j)), w.ones_h);
    }
    const norms = if (lay.global) [3]Array{ lay.norm_q, lay.norm_k, lay.norm_v } else [3]Array{ lay.norm_q, lay.norm_q, lay.norm_q };
    var ins: [3]Array = undefined;
    for (&ins, xs, norms) |*in, x, nw| in.* = try g.astype(try rms(g, x, nw, eps), dt);

    const a = try attention.forward(&lay.attn, c, ins[0], ins[1], ins[2], prev.k, prev.v);
    const pa = try conv.shortConv(g, a.y, lay.post_attn_conv, prev.attn);
    const xa = try g.astype(pa.y, .float32);
    var bs_next = bs;
    var res = xa;
    if (lay.boundary) {
        const b = try g.expandDims(nr, 0);
        bs_next = if (bs) |old| try g.concat(&.{ old, b }, 0) else b;
    } else {
        res = try g.add(try g.astype(br, .float32), xa);
    }
    var mi = try depth.row(g, try depth.attend(g, bs_next, try urms(g, res, w.ones_h), lay.mlp_query), 0);
    if (msgs) |m| mi = try depth.gate(g, &lay.gates.?[3], mi, try depth.row(g, m, 3), w.ones_h);
    const mo = try moe.forward(&lay.moe, g, try g.astype(try rms(g, mi, lay.post_attn_norm, eps), dt));
    const pm = try conv.shortConv(g, mo, lay.post_mlp_conv, prev.mlp);
    // The concat is a fresh copy: the state never pins the chunk.
    try kv.setState(c, .conv, try g.concat(&.{ a.k_tail, a.v_tail, pa.tail, pm.tail }, 2));
    if (c.capture_spec) try kv.setState(c, .conv_input, try g.concat(&.{ a.k_window, a.v_window, pa.window, pm.window }, 2));
    return .{ .h = try g.astype(try g.add(res, try g.astype(pm.y, .float32)), dt), .bs = bs_next };
}

/// The activation dtype: what `mx.dequantize` returns for the embedding table by default (the affine scales' dtype;
/// bf16 for fp-mode tables, whose scales are uint8).
pub fn activationDtype(table: *const Linear) mlx.mlx_dtype {
    const dt = ops.dtype(if (table.s.ctx != null) table.s else table.w);
    return switch (dt) {
        .bfloat16, .float16, .float32 => dt,
        else => .bfloat16,
    };
}

/// Replaces the graph-owned handle `slot` by a new reference on `x` (or null), dropping the old one.
fn carry(g: *Graph, slot: *?Array, x: ?Array) !void {
    const next = if (x) |a| try g.keep(a) else null;
    if (slot.*) |old| g.drop(old);
    slot.* = next;
}

pub fn forward(w: *const Weights, shared: *const block.Shared, g: *Graph, ids: Array, run: run_mod.Run) !run_mod.Out {
    if (run.mode == .batched_rows) return error.BatchedRowsUnsupported;
    var c = run.ctx(g, shared);
    const dt = activationDtype(&w.embed);
    const x = try embed.gatherRows(g, &w.embed, ids, dt);
    var h: ?Array = try g.astype(try rms(g, x, w.embed_norm, w.eps), dt);
    var bs: ?Array = null;
    var snap_h: ?Array = null;
    var snap_bs: ?Array = null;
    var bank_blocks: ?Array = null;
    var bank_partial: ?Array = null;
    const n = w.visits.len;
    const core_end = w.core_start + w.core_len * w.passes;
    const d: c_int = @intCast(w.hidden);
    for (0..n) |vi| {
        const v: u32 = @intCast(vi);
        const i: u32 = w.visits[v];
        const in_core = w.core_len > 0 and v >= w.core_start and v < core_end;
        const pass: u32 = if (in_core) (v - w.core_start) / w.core_len else 0;
        if (in_core and v == w.core_start and w.passes > 1) {
            try carry(g, &snap_h, h);
            try carry(g, &snap_bs, bs);
        }
        if (in_core and pass > 0 and i == w.core_start) {
            // The bank: the blocks this pass opened after the prelude's, and the normed hidden; then back to the prelude.
            var s = g.scope();
            defer s.deinit();
            const skip: c_int = @intCast(w.n_prelude_blocks + 1);
            const nb: c_int = if (bs) |b| ops.shape(b)[0] else 0;
            try carry(g, &bank_blocks, if (nb > skip) try s.sliceAxis(bs.?, 0, skip, nb) else null);
            try carry(g, &bank_partial, try urms(&s, try s.astype(h.?, .float32), w.ones_h));
            try carry(g, &h, snap_h);
            try carry(g, &bs, snap_bs);
        }
        // The hidden entering visit v: what a drafter's capture of this visit's input reads.
        const entering = h.?;
        var t = g.scope();
        defer t.deinit();
        c.g = &t;
        c.layer = v;
        const msgs: ?Array = if (in_core and pass > 0) blk: {
            const j: c_int = @intCast(i - w.core_start);
            const q = try t.reshape(try t.sliceAxis(w.message_query, 0, j, j + 1), &.{ 4, d });
            break :blk try depth.attend(&t, bank_blocks, bank_partial.?, q);
        } else null;
        const out = try visit(w, &w.layers[i], &c, entering, bs, msgs, dt);
        c.g = g;
        try carry(g, &h, out.h);
        if (out.bs) |nb| if (bs == null or nb.ctx != bs.?.ctx) try carry(g, &bs, nb);
        try run.capture(g, v, h.?);
        try run.cadence(vi, n, h.?);
    }
    const fin = try depth.row(g, try depth.attend(g, bs, try urms(g, try g.astype(h.?, .float32), w.ones_h), w.final_query), 0);
    const final = try g.astype(try rms(g, fin, w.norm, w.eps), dt);
    return run.finish(g, &c, final, Head, w);
}

const testReadF32 = @import("../../sdk/testing/parity.zig").testReadF32;

test "inkling2 trunk: an fp-mode embedding table gathers in bf16" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(std.testing.allocator, s);
    defer g.deinit();
    // mxfp8 g32, 4 rows of 32: every code 0x38 (e4m3 1.0), every scale 127 (e8m0 1.0).
    const codes: [32]u32 = @splat(0x38383838);
    const scales: [4]u8 = @splat(127);
    const table = Linear{
        .w = try g.fromSlice(u32, &codes, &.{ 4, 8 }),
        .s = try g.adopt(mlx.mlx_array_new_data(&scales, &[_]c_int{ 4, 1 }, 2, .uint8)),
        .bits = 8,
        .group_size = 32,
        .mode = .mxfp8,
    };
    const dt = activationDtype(&table);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, dt);
    const rows = try embed.gatherRows(&g, &table, try g.fromSlice(i32, &.{ 2, 0 }, &.{2}), dt);
    var got: [2 * 32]f32 = undefined;
    try testReadF32(rows, &got, s);
    for (got) |v| try std.testing.expectEqual(@as(f32, 1.0), v);
}
