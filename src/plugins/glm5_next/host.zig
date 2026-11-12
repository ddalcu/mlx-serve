//! The table glue both KDA-hybrid plugins share: load and unload of the trunk (`trunk.zig`), the trunk weights a
//! drafter reads, a forward's token ids, one slot's rows through a table forward module and its head into
//! `ForwardOut`. Per-request state lives in the slots' caches, never here.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const quant_abi = @import("../../abi/quant.zig");
const Layout = @import("../../sdk/quant/native.zig").Layout;
const trunk = @import("trunk.zig");
const log = @import("../../sdk/util/log.zig");

const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;

pub const Model = trunk.Model;

pub fn load(comptime name: []const u8, ctx: *const abi.LoadCtx, out: *abi.Verdict) ?*anyopaque {
    return trunk.load(ctx) catch |e| {
        out.* = switch (e) {
            error.OutOfMemory => abi.Verdict.refuse(.failed, name ++ ": out of memory"),
            // The driver maps this name to the engine's own refusal.
            error.MissingWeight => abi.Verdict.refuse(.failed, "MissingWeight"),
            error.UnsupportedWeightSource => abi.Verdict.refuse(.unsupported, name ++ ": a weight source other than the resident map"),
            else => abi.Verdict.named(.failed, @errorName(e)),
        };
        return null;
    };
}

pub fn unload(mp: *anyopaque) callconv(.c) void {
    const m = of(mp);
    m.deinit();
    gpa.destroy(m);
}

pub fn of(mp: *anyopaque) *Model {
    return @ptrCast(@alignCast(mp));
}

/// `SpecHooks.trunk_weight`: the loaded embedding table or head, borrowed (a tied head is the table).
pub fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) abi.Verdict {
    const m = of(mp);
    const l = switch (which) {
        .embed => &m.embed,
        // A head the outside plugin runs lends its draft readout (`trunk.draftReadout`) where one was built.
        .head => if (m.head.ext != null and m.draft_head.w.ctx != null) &m.draft_head else &m.head,
    };
    if (l.w.ctx == null or l.ext != null) return abi.Verdict.refuse(.unsupported, "the trunk has no such weight");
    out.* = .{ .w = .{ .ctx = l.w.ctx }, .scales = .{ .ctx = l.s.ctx }, .biases = .{ .ctx = l.b.ctx } };
    if (l.s.ctx == null) return .ok;
    out.bits = @intCast(l.bits);
    out.group_size = l.group_size;
    out.layout = @backingInt(switch (l.mode) {
        .nvfp4 => Layout.nvfp4,
        .mxfp4 => Layout.mxfp4,
        .mxfp8 => Layout.mxfp8,
        else => Layout.affine,
    });
    return .ok;
}

/// The forward's token ids `[1, rows]`: the engine's device array, or the host tokens as int32.
pub fn tokenIds(f: *const abi.Forward) !Array {
    if (f.ids.ctx != null) {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, .{ .ctx = f.ids.ctx });
        return r;
    }
    return hostIds(f.tokens.slice());
}

/// Host tokens as an int32 `[1, n]` window (MLX copies them).
pub fn hostIds(tokens: []const u32) !Array {
    const buf = try gpa.alloc(i32, tokens.len);
    defer gpa.free(buf);
    for (tokens, buf) |t, *b| b.* = std.math.cast(i32, t) orelse return error.TokenOutOfRange;
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(tokens.len) }, 2, .int32);
}

/// The rows a `Caps.prompt_last_row` model's head reads of a last-row prompt (`finish`): past the 16 rows a decode call
/// takes, so the head's matmul runs as the prompt's.
pub const head_tail_rows: c_int = 32;

/// Fills `out` from the final-normed hidden `[1, S, H]` (consumed): the hidden when asked, and the logits the
/// forward wants (`[S, V]` or the last row `[1, V]`).
pub fn finish(m: *Model, f: *const abi.Forward, hidden: Array, out: *abi.ForwardOut) !void {
    defer _ = mlx.mlx_array_free(hidden);
    if (f.want_hidden) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        out.hidden = .{ .ctx = h.ctx };
    }
    if (f.logits == .none) return;
    const s = m.s;
    // A model that declares `Caps.prompt_last_row` (bailing_hybrid's streaming pack) runs a last-row prompt's head over
    // its last `head_tail_rows` rows only: still a prompt call, whose rows its outside plugin's matmul computes as the
    // whole prompt's call does, so the last row keeps its `.all_rows` bytes.
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    const hsh = mlx.getShape(hidden);
    const head_in = if (f.logits == .last_row and m.caps.prompt_last_row and hsh[1] > head_tail_rows) blk: {
        try mlx.check(mlx.mlx_slice(&tail, hidden, &[_]c_int{ 0, hsh[1] - head_tail_rows, 0 }, 3, &[_]c_int{ 1, hsh[1], hsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        break :blk tail;
    } else hidden;
    const logits = try m.lmHead(head_in);
    defer _ = mlx.mlx_array_free(logits);
    const sh = mlx.getShape(logits);
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    if (f.logits == .all_rows) {
        try mlx.check(mlx.mlx_reshape(&rows, logits, &[_]c_int{ sh[1], sh[2] }, 2, s));
    } else {
        var last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(last);
        try mlx.check(mlx.mlx_slice(&last, logits, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_reshape(&rows, last, &[_]c_int{ 1, sh[2] }, 2, s));
    }
    out.logits = .{ .ctx = rows.ctx };
}

/// Frees what a failed forward left in `out`.
pub fn clear(out: *abi.ForwardOut, captured: usize) void {
    for ([_]*abi.Array{ &out.logits, &out.hidden }) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
        a.* = .{};
    };
    if (out.captured) |c| for (c[0..captured]) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(.{ .ctx = a.ctx });
        a.* = .{};
    };
}

/// One slot's rows through `fwd.hidden` (a table forward module), its head and captures into `out`; a split prompt
/// forward (`Forward.splitRows`) through `forwardSplit`.
pub fn forwardOne(comptime name: []const u8, comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    if (f.splitRows() > 0) return forwardSplit(name, fwd, m, ops, f, out);
    if (f.caches.len != 1 or f.positions.len != 1) return abi.Verdict.refuse(.unsupported, name ++ ": one slot's rows at a time");
    if (f.capture_layers.len > 0 and out.captured == null) return abi.Verdict.refuse(.failed, name ++ ": capture layers without an output");
    if (f.media.ctx != null and !@hasField(fwd.Step, "media")) return abi.Verdict.refuse(.unsupported, name ++ ": no media rows");
    one(fwd, m, ops, f, out) catch |e| {
        clear(out, f.capture_layers.len);
        return abi.Verdict.named(.failed, @errorName(e));
    };
    return .ok;
}

fn one(comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const ids = try tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    var caps: [64]Array = undefined;
    const n_cap = f.capture_layers.len;
    if (n_cap > caps.len) return error.TooManyCaptureLayers;
    for (caps[0..n_cap]) |*c| c.* = mlx.mlx_array_new();
    defer for (caps[0..n_cap]) |c| {
        _ = mlx.mlx_array_free(c);
    };
    var st: fwd.Step = .{
        .slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = f.positions.slice() },
        // A verify's rows capture their recurrent states for a partial accept's rollback.
        .capture = f.mode == .verify,
        .capture_ids = f.capture_layers.slice(),
        .capture_out = caps[0..n_cap],
        .eval_every = f.eval_every,
    };
    if (@hasField(fwd.Step, "media")) {
        st.media = .{ .ctx = f.media.ctx };
        st.media_offset = f.media_offset;
    }
    try finish(m, f, try fwd.hidden(m, &st, ids), out);
    for (caps[0..n_cap], 0..) |c, i| {
        var r = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&r, c);
        out.captured.?[i] = .{ .ctx = r.ctx };
    }
}

/// Why a split prompt forward is refused, by name and logged, before any cache is touched (the host then runs two
/// forwards); null when it is served or the forward is not split. A caller with per-forward bookkeeping asks this first.
pub fn splitRefusal(comptime name: []const u8, comptime fwd: type, m: *const Model, f: *const abi.Forward) ?abi.Verdict {
    if (f.splitRows() == 0) return null;
    if (comptime !@hasDecl(fwd, "hiddenSplit")) {
        return abi.Verdict.refuse(.unsupported, name ++ ": no split prompt forward");
    } else {
        _ = splitSpansOf(name, fwd, m, f) catch |e| return abi.Verdict.named(.unsupported, @errorName(e));
        return null;
    }
}

/// The split's spans (`fwd.splitSpans` over `fwd.splitServes`), or its refusal, logged.
fn splitSpansOf(comptime name: []const u8, comptime fwd: type, m: *const Model, f: *const abi.Forward) fwd.SplitRefusal!fwd.Spans {
    const rows_a = f.splitRows();
    const rows_b = f.rows -| rows_a;
    return fwd.splitSpans(f, rows_a, f.splitStates() != null, fwd.splitServes(m, rows_a, rows_b)) catch |e| {
        log.info("[" ++ name ++ "] a split prompt forward of {d} + {d} rows is refused: {s}\n", .{ rows_a, rows_b, @errorName(e) });
        return e;
    };
}

/// A split prompt forward: span A's rows then span B's on the one slot's caches through `fwd.hiddenSplit` (layer by
/// layer, both spans' routed experts in one streamed call a layer, span A's recurrent states into `Forward.split_states`);
/// `out.logits` what a forward of span B alone returns, `out.hidden` span A's rows then span B's, each as its own forward
/// returns them. Refused by name before any cache is touched where the spans would not be their own forwards'.
fn forwardSplit(comptime name: []const u8, comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    if (comptime !@hasDecl(fwd, "hiddenSplit")) {
        return abi.Verdict.refuse(.unsupported, name ++ ": no split prompt forward");
    } else {
        const spans = splitSpansOf(name, fwd, m, f) catch |e| return abi.Verdict.named(.unsupported, @errorName(e));
        splitOne(fwd, m, ops, f, spans, out) catch |e| {
            clear(out, 0);
            return abi.Verdict.named(.failed, @errorName(e));
        };
        return .ok;
    }
}

fn splitOne(comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, sp: fwd.Spans, out: *abi.ForwardOut) !void {
    const ids_a = try spanIds(m, f, 0, sp.a);
    defer _ = mlx.mlx_array_free(ids_a);
    const ids_b = try spanIds(m, f, sp.a, sp.a + sp.b);
    defer _ = mlx.mlx_array_free(ids_b);
    // Each span's step as `one` makes its forward's (a prompt's: no capture), span B's from span A's end.
    const pos_a = [1]u32{sp.pos_a};
    const pos_b = [1]u32{sp.pos_b};
    const st_a: fwd.Step = .{ .slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = &pos_a }, .eval_every = f.eval_every };
    const st_b: fwd.Step = .{ .slots = .{ .ops = ops, .caches = f.caches.slice(), .positions = &pos_b }, .eval_every = f.eval_every };
    const h = try fwd.hiddenSplit(m, &st_a, &st_b, ids_a, ids_b, f.splitStates());
    try finishSplit(m, f, h[0], h[1], out);
}

/// Rows `lo..hi` of a forward's token ids `[1, hi - lo]`: the host tokens as `hostIds` makes a forward's, or a view of
/// the engine's device ids `[1, rows]`.
fn spanIds(m: *const Model, f: *const abi.Forward, lo: u32, hi: u32) !Array {
    if (f.ids.ctx == null) return hostIds(f.tokens.slice()[lo..hi]);
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_slice(&r, .{ .ctx = f.ids.ctx }, &[_]c_int{ 0, @intCast(lo) }, 2, &[_]c_int{ 1, @intCast(hi) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
    return r;
}

/// `out` of a split forward from both spans' final-normed hiddens (consumed): the logits `finish` gives a forward of
/// span B alone over span B's, and, when asked, the hidden of span A's rows then span B's.
fn finishSplit(m: *Model, f: *const abi.Forward, hidden_a: Array, hidden_b: Array, out: *abi.ForwardOut) !void {
    defer _ = mlx.mlx_array_free(hidden_a);
    if (f.want_hidden) {
        const parts = [_]Array{ hidden_a, hidden_b };
        const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var h = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&h, vec, 1, m.s)) catch |e| {
            _ = mlx.mlx_array_free(h);
            _ = mlx.mlx_array_free(hidden_b);
            return e;
        };
        out.hidden = .{ .ctx = h.ctx };
    }
    // `finish` reads the forward's `logits` and `want_hidden` only: span B's forward, its hidden already in `out`.
    var fb = f.*;
    fb.want_hidden = false;
    try finish(m, &fb, hidden_b, out);
}

/// The request's summary lines (`ModelTable.meters`): a streaming pack's bank's (`[stream-stats]`); none otherwise.
pub fn meters(mp: *anyopaque, _: u32, out: *abi.Str) callconv(.c) void {
    const m = of(mp);
    out.* = if (m.bank) |b| abi.Str.of(b.meters()) else .{};
}

/// A streaming pack's bank after a forward: its outputs evaluated, then the calls' slot rows settled for the next
/// forward's reads.
pub fn settleBank(m: *Model, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    const r = m.routed orelse return .ok;
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for ([_]abi.Array{ out.logits, out.hidden }) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, .{ .ctx = a.ctx });
    };
    if (out.captured) |c| for (c[0..f.capture_layers.len]) |a| if (a.ctx != null) {
        _ = mlx.mlx_vector_array_append_value(vec, .{ .ctx = a.ctx });
    };
    mlx.check(mlx.mlx_eval(vec)) catch |e| {
        clear(out, f.capture_layers.len);
        return abi.Verdict.named(.failed, @errorName(e));
    };
    r.settle() catch |e| {
        clear(out, f.capture_layers.len);
        return abi.Verdict.named(.failed, @errorName(e));
    };
    return .ok;
}
