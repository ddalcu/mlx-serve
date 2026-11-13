//! Ling 3.0's forward behind the model table: KDA layers on the slot's recurrent state and MLA layers on its KV, both
//! through `CacheOps`, then the MoE (or the dense first layer). One slot's rows per call; a verify captures its KDA
//! rows. `hiddenSpans` runs a whole prompt's spans layer by layer, each span's bytes `hidden`'s.
//! copied_from: engines/mlx/forward/layered.zig (`forwardMoeWith`, its bailing arms).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const slots_mod = @import("../glm5_next/slots.zig");
const trunk = @import("../glm5_next/trunk.zig");
const glm_moe = @import("../glm5_next/moe.zig");
const moe_block = @import("../../sdk/bases/blocks/moe.zig");
const mla = @import("mla.zig");
const span_plan = @import("span_plan.zig");

const Array = mlx.mlx_array;

pub const Step = struct {
    slots: slots_mod.Slots,
    capture: bool = false,
    capture_ids: []const u32 = &.{},
    capture_out: []Array = &.{},
    eval_every: u32 = 0,
};

/// The final-normed hidden `[1, S, H]`; the head is the caller's.
pub fn hidden(m: *trunk.Model, st: *const Step, token_ids: Array) !Array {
    const offset: c_int = @intCast(st.slots.positions[0]);
    var h = try m.embedding(token_ids);
    defer _ = mlx.mlx_array_free(h);
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    // The previous layer's tail may already have produced this layer's input norm.
    var carried_normed: ?Array = null;
    errdefer if (carried_normed) |c| {
        _ = mlx.mlx_array_free(c);
    };
    const n_layers = m.layers.len;
    for (m.layers, 0..) |*lw, layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const normed = if (carried_normed) |c| c else try m.rmsNorm(h, lw.input_norm);
        carried_normed = null;
        defer _ = mlx.mlx_array_free(normed);

        const attn_out = try mixer(m, st.slots, lw, li, normed, offset, batch, seq_len, st.capture);
        defer _ = mlx.mlx_array_free(attn_out);
        const ff_normed = (try residual(m, &h, attn_out, lw.post_norm, true)).?;
        defer _ = mlx.mlx_array_free(ff_normed);
        const mlp_out = switch (lw.mlp) {
            .moe => |*mw| if (m.serial_rows) try glm_moe.forwardRows(mw, m.s, ff_normed) else try glm_moe.forward(mw, m.s, ff_normed),
            .dense => |*dw| try m.denseMlp(dw, ff_normed),
        };
        defer _ = mlx.mlx_array_free(mlp_out);

        const last = layer_idx + 1 == n_layers;
        const next_norm = if (last) m.final_norm else m.layers[layer_idx + 1].input_norm;
        carried_normed = try residual(m, &h, mlp_out, next_norm, false);

        // DFlash taps `hidden_states[li + 1]`, the layer's output.
        for (st.capture_ids, st.capture_out) |cid, *slot| if (cid == li) {
            _ = mlx.mlx_array_set(slot, h);
        };
        if (is_prefill and st.eval_every > 0 and ((layer_idx + 1) % st.eval_every == 0 or last)) {
            try st.slots.evalWith(h, li + 1);
        }
    }
    if (carried_normed) |c| {
        carried_normed = null;
        return c;
    }
    return m.rmsNorm(h, m.final_norm);
}

/// Layer `li`'s mixer over slot 0's rows `x` at `offset`.
fn mixer(m: *trunk.Model, sl: slots_mod.Slots, lw: *const trunk.Layer, li: u32, x: Array, offset: c_int, batch: c_int, seq_len: c_int, capture: bool) !Array {
    const is_prefill = seq_len > 1;
    return switch (lw.mixer) {
        .kda => |*la| slots_mod.kda(m, sl, 0, x, la, li, batch, seq_len, is_prefill, capture),
        .mla => |*w| mla.mlaAttn(m, sl, x, w, li, offset, batch, seq_len, is_prefill),
        .dsa => error.LingLayerIsDsa,
    };
}

/// `h + delta` into `h`, and its norm by `w`: one kernel where one serves (decode and verify widths), else the add,
/// then the norm only when `norm` (null otherwise).
fn residual(m: *trunk.Model, h: *Array, delta: Array, w: Array, norm: bool) !?Array {
    if (try m.narrowAddNorm(h.*, delta, w)) |fused| {
        _ = mlx.mlx_array_free(h.*);
        h.* = fused.h;
        return fused.normed;
    }
    var h_new = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&h_new, h.*, delta, m.s));
    _ = mlx.mlx_array_free(h.*);
    h.* = h_new;
    return if (norm) try m.rmsNorm(h.*, w) else null;
}

/// Each KDA layer's conv window and scan state after the spans `after` (ascending, the pass's span indices), into
/// `conv[c][layer]` and `ssm[c][layer]`: references the caller frees.
pub const Keep = struct {
    after: []const usize = &.{},
    conv: []const []Array = &.{},
    ssm: []const []Array = &.{},
};

/// A whole prompt's spans for `hiddenSpans`: slot 0's cache with the first span's position, each span's rows in order.
pub const Spans = struct {
    slots: slots_mod.Slots,
    rows: []const u32,
    keep: Keep = .{},
    /// Layers whose output (every span's rows, joined) goes into `capture_out`, as `Step`'s.
    capture_ids: []const u32 = &.{},
    capture_out: []Array = &.{},
    /// Rows the engine's gauge reads (`Forward.progress`): `done` before these spans, then their share after each layer.
    progress: ?*u64 = null,
    done: u64 = 0,
};

/// `sp`'s spans (token ids `token_ids` `[1, rows]`) layer by layer, each span's bytes what `hidden` gives it as a forward
/// of its own after the spans before it, as the engine's chunk loop runs them. Every op of a span runs on that span's
/// rows alone, at its position, in the spans' order (embedding, norms, mixer on the slot's cache, residuals, dense
/// MLP, router, shared expert), but a MoE layer's routed experts: one call over whole spans (`span_plan.callEnd` at
/// `Model.prompt_calls`), whose prompt rows each carry the bytes a call of their span alone gives them, so a layer reads
/// each expert once a call rather than once a span. Spans of a decode lane's rows or fewer are the caller's. Each
/// span's final-normed hidden `[1, S, H]` into `out` (the caller frees them).
pub fn hiddenSpans(m: *trunk.Model, sp: *const Spans, token_ids: Array, out: []Array) !void {
    const a = std.heap.c_allocator;
    const n = sp.rows.len;
    if (n == 0 or out.len != n) return error.SpanMajorSpans;
    const s = m.s;
    const starts = try a.alloc(u32, n + 1);
    defer a.free(starts);
    starts[0] = 0;
    for (sp.rows, 0..) |r, j| starts[j + 1] = starts[j] + r;
    const pos = try a.alloc(u32, n);
    defer a.free(pos);
    for (pos, starts[0..n]) |*p, st| p.* = sp.slots.positions[0] + st;
    const h = try a.alloc(Array, n);
    defer a.free(h);
    const carried = try a.alloc(Array, n);
    defer a.free(carried);
    const ff = try a.alloc(Array, n);
    defer a.free(ff);
    const mlp_out = try a.alloc(Array, n);
    defer a.free(mlp_out);
    inline for (.{ h, carried, ff, mlp_out }) |xs| @memset(xs, .{ .ctx = null });
    defer inline for (.{ h, carried, ff, mlp_out }) |xs| freeAll(xs);
    for (h, 0..) |*x, j| {
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_slice(&ids, token_ids, &[_]c_int{ 0, @intCast(starts[j]) }, 2, &[_]c_int{ 1, @intCast(starts[j + 1]) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        x.* = try m.embedding(ids);
    }
    const n_layers = m.layers.len;
    for (m.layers, 0..) |*lw, layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const last = layer_idx + 1 == n_layers;
        const next_norm = if (last) m.final_norm else m.layers[layer_idx + 1].input_norm;
        for (0..n) |j| {
            const span_sl: slots_mod.Slots = .{ .ops = sp.slots.ops, .caches = sp.slots.caches, .positions = pos[j .. j + 1] };
            const normed = if (carried[j].ctx != null) carried[j] else try m.rmsNorm(h[j], lw.input_norm);
            carried[j] = .{ .ctx = null };
            defer _ = mlx.mlx_array_free(normed);
            const attn_out = try mixer(m, span_sl, lw, li, normed, @intCast(pos[j]), 1, @intCast(sp.rows[j]), false);
            defer _ = mlx.mlx_array_free(attn_out);
            ff[j] = (try residual(m, &h[j], attn_out, lw.post_norm, true)).?;
            switch (lw.mixer) {
                .kda => {
                    for (sp.keep.after, 0..) |aj, c| if (aj == j) {
                        sp.keep.conv[c][layer_idx] = try kept(span_sl.state(0, li, .conv), s);
                        sp.keep.ssm[c][layer_idx] = try kept(span_sl.state(0, li, .ssm), s);
                    };
                },
                // A span's attention evaluated before the next span's append, as the chunk loop's chunk is: one score
                // sheet at a time, and the cache's next write updates its buffer in place.
                .mla => try evalAll(&.{ h[j], ff[j] }),
                .dsa => {},
            }
        }
        switch (lw.mlp) {
            .dense => |*dw| {
                for (ff, mlp_out) |x, *y| y.* = try m.denseMlp(dw, x);
            },
            .moe => |*mw| try moeSpans(m, mw, ff, sp.rows, mlp_out),
        }
        for (h, carried, mlp_out) |*x, *c, y| c.* = (try residual(m, x, y, next_norm, false)) orelse .{ .ctx = null };
        freeAll(ff);
        freeAll(mlp_out);
        for (sp.capture_ids, sp.capture_out) |cid, *slot| if (cid == li) {
            const joined = try join(h, s);
            defer _ = mlx.mlx_array_free(joined);
            _ = mlx.mlx_array_set(slot, joined);
        };
        try evalLayer(sp.slots, li, lw, h, carried);
        if (sp.progress) |p| @atomicStore(u64, p, sp.done + @as(u64, starts[n]) * @as(u64, layer_idx + 1) / @as(u64, n_layers), .monotonic);
    }
    for (out, h, carried) |*o, x, *c| {
        if (c.ctx != null) {
            o.* = c.*;
            c.* = .{ .ctx = null };
        } else o.* = try m.rmsNorm(x, m.final_norm);
    }
}

/// A MoE layer over every span's normed rows `ff`: each span routed alone (`mlpRoute`), the routed experts in calls over
/// whole spans (`routedSum`), then each span's shared expert and add on its rows of its call (`mlpFinish`).
fn moeSpans(m: *trunk.Model, mw: *const moe_block.Moe.Weights, ff: []const Array, rows: []const u32, out: []Array) !void {
    const a = std.heap.c_allocator;
    var g = ops.Graph.init(a, m.s);
    defer g.deinit();
    const n = rows.len;
    const inds = try a.alloc(Array, n);
    defer a.free(inds);
    const scores = try a.alloc(Array, n);
    defer a.free(scores);
    for (ff, inds, scores) |x, *i, *w| {
        const r = try moe_block.mlpRoute(mw, &g, x);
        i.* = r.inds;
        w.* = r.scores;
    }
    var j0: usize = 0;
    while (j0 < n) {
        const j1 = span_plan.callEnd(rows, j0, m.prompt_calls.rows, m.prompt_calls.tail);
        const one = j1 - j0 == 1;
        const x = if (one) ff[j0] else try g.concat(ff[j0..j1], 1);
        const sum = try moe_block.routedSum(mw, &g, x, if (one) inds[j0] else try g.concat(inds[j0..j1], 1), if (one) scores[j0] else try g.concat(scores[j0..j1], 1));
        var at: c_int = 0;
        for (j0..j1) |j| {
            const r: c_int = @intCast(rows[j]);
            const part = if (one) sum else try g.sliceAxis(sum, 1, at, at + r);
            out[j] = g.escape(try moe_block.mlpFinish(mw, &g, ff[j], part));
            at += r;
        }
        j0 = j1;
    }
}

/// Every span's rows `[1, S, H]` joined along the rows (a new reference).
fn join(xs: []const Array, s: mlx.mlx_stream) !Array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    if (xs.len == 1) {
        _ = mlx.mlx_array_set(&r, xs[0]);
        return r;
    }
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, 1, s));
    return r;
}

/// A layer's outputs evaluated with its KDA state (an MLA layer's cache is each span's), so the next layer's graph holds
/// none of this one's.
fn evalLayer(sl: slots_mod.Slots, li: u32, lw: *const trunk.Layer, h: []const Array, carried: []const Array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (h) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
    for (carried) |x| if (x.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, x);
    };
    if (lw.mixer == .kda) for ([_]abi.StateSlot{ .conv, .ssm }) |which| {
        const st = sl.state(0, li, which);
        if (st.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, st);
    };
    try mlx.check(mlx.mlx_eval(vec));
}

fn evalAll(xs: []const Array) !void {
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// A recurrent state kept for a checkpoint: its own buffer, evaluated, as the engine files a checkpoint's copy
/// (`materializedOwnedCopy`). The live state may be a slice of its span's conv input, which a reference would keep.
pub fn kept(x: Array, s: mlx.mlx_stream) !Array {
    if (x.ctx == null) return mlx.mlx_array_new();
    const c = try materializedOwnedCopy(s, x);
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

fn freeAll(xs: []Array) void {
    for (xs) |*x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x.*);
        x.* = .{ .ctx = null };
    };
}
