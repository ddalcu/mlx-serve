//! The table glue both KDA-hybrid plugins share: load and unload of the trunk (`trunk.zig`), the trunk weights a
//! drafter reads, a forward's token ids, one slot's rows through a table forward module and its head into
//! `ForwardOut`. Per-request state lives in the slots' caches, never here.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const quant_abi = @import("../../../abi/quant.zig");
const Layout = @import("../../quant/layout.zig").Layout;
const trunk = @import("trunk.zig");

const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;

pub const Model = trunk.Model;

pub fn load(comptime name: []const u8, ctx: *const abi.LoadCtx, hooks: trunk.Hooks, out: *abi.Verdict) ?*anyopaque {
    return trunk.load(ctx, hooks) catch |e| {
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

/// Fills `out` from the final-normed hidden `[1, S, H]` (consumed): the hidden when asked, and the logits the
/// forward wants (`[S, V]`, or the last row `[1, V]` sliced from the same head call over every row).
pub fn finish(m: *Model, f: *const abi.Forward, hidden: Array, out: *abi.ForwardOut) !void {
    defer _ = mlx.mlx_array_free(hidden);
    if (f.want_hidden) {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, hidden);
        out.hidden = .{ .ctx = h.ctx };
    }
    if (f.logits == .none) return;
    const s = m.s;
    const logits = try m.lmHead(hidden);
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

/// One slot's rows through `fwd.hidden` (a table forward module), its head and captures into `out`.
pub fn forwardOne(comptime name: []const u8, comptime fwd: type, m: *Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
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

// Bar: on the GPU, a last-row ask returns the bytes of the all-rows ask's last row through a dense bf16 head of K 2304
// and N 8192 at 65 rows, where MLX's matmul takes another kernel at 32 rows than at 65.
test "kda_hybrid finish: a last-row prompt serves the all-rows head's last row bit for bit (dense bf16 head, 65 rows)" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const testing = std.testing;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const k: c_int = 2304;
    const n_out: c_int = 8192;
    const rows: c_int = 65;
    var prng = std.Random.DefaultPrng.init(65);
    const wbuf = try testing.allocator.alloc(f32, @intCast(n_out * k));
    defer testing.allocator.free(wbuf);
    for (wbuf) |*x| x.* = prng.random().floatNorm(f32) * 0.02;
    const xbuf = try testing.allocator.alloc(f32, @intCast(rows * k));
    defer testing.allocator.free(xbuf);
    for (xbuf) |*x| x.* = prng.random().floatNorm(f32);
    const wf = mlx.mlx_array_new_data(wbuf.ptr, &[_]c_int{ n_out, k }, 2, .float32);
    defer _ = mlx.mlx_array_free(wf);
    const xf = mlx.mlx_array_new_data(xbuf.ptr, &[_]c_int{ 1, rows, k }, 3, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_astype(&w, wf, .bfloat16, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    var m: Model = .{ .c = undefined, .s = s, .caps = .{ .prompt_last_row = true }, .embed = .{}, .head = .{ .w = w }, .final_norm = .{} };
    var got: [2][]f32 = undefined;
    for ([_]abi.Logits{ .all_rows, .last_row }, 0..) |mode, i| {
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, x);
        var out: abi.ForwardOut = .{};
        try finish(&m, &.{ .mode = .prefill, .logits = mode, .rows = @intCast(rows), .tokens = .{} }, h, &out);
        const l: Array = .{ .ctx = out.logits.ctx };
        defer _ = mlx.mlx_array_free(l);
        var f32s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32s);
        try mlx.check(mlx.mlx_astype(&f32s, l, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f32s));
        const sh = mlx.getShape(f32s);
        const last = mlx.mlx_array_data_float32(f32s).?[@intCast((sh[0] - 1) * sh[1])..@intCast(sh[0] * sh[1])];
        got[i] = try testing.allocator.dupe(f32, last);
    }
    defer for (got) |g| testing.allocator.free(g);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[0]), std.mem.sliceAsBytes(got[1]));
}
