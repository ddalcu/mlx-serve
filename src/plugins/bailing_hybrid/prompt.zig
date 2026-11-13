//! bailing_hybrid's whole-prompt forward (`Caps.whole_prompt_prefill`, a streaming pack): the engine's chunk loop spans
//! (`Forward.prompt_spans`) layer by layer (`table_forward.hiddenSpans`), so a routed layer reads each expert once a
//! call rather than once a span, every span's bytes the chunk loop's. A span the bank's decode lane holds runs as the
//! chunk loop's forward of its own (its phase, routes and settle). Each KDA layer's state after a span ending at a
//! checkpoint position goes to the engine (`CacheOps.checkpoint`) once every layer has passed it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const host = @import("../glm5_next/host.zig");
const trunk = @import("../glm5_next/trunk.zig");
const fwd = @import("table_forward.zig");
const span_plan = @import("span_plan.zig");
const name = @import("config.zig").plugin.name;

const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;
const Parts = std.ArrayList(Array);
const max_captures = 64;
const max_checkpoints = 64;

/// The prompt `f` (a prefill naming its spans) into `out`: the logits it asks for, its hidden rows and captures.
pub fn forward(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut) abi.Verdict {
    var why: abi.Verdict = .ok;
    run(m, ops, f, out, &why) catch |e| {
        host.clear(out, f.capture_layers.len);
        return if (why.isOk()) abi.Verdict.named(.failed, @errorName(e)) else why;
    };
    return .ok;
}

const Pass = struct {
    m: *trunk.Model,
    ops: *const abi.CacheOps,
    f: *const abi.Forward,
    /// Each checkpoint's KDA states by layer (`ssm_checkpoints`' order), filled where its span ends.
    conv: [][]Array,
    ssm: [][]Array,
    logits: Parts = .empty,
    hidden: Parts = .empty,
    captures: [max_captures]Parts = @splat(Parts.empty),

    fn deinit(p: *Pass) void {
        freeParts(&p.logits);
        freeParts(&p.hidden);
        for (&p.captures) |*c| freeParts(c);
    }

    fn slots(p: *const Pass, pos: *const [1]u32) @FieldType(fwd.Spans, "slots") {
        return .{ .ops = p.ops, .caches = p.f.caches.slice(), .positions = pos };
    }

    /// What a span's forward made, moved into the pass's parts (`so` left empty).
    fn take(p: *Pass, so: *abi.ForwardOut) !void {
        if (so.logits.ctx != null) {
            try p.logits.append(gpa, .{ .ctx = so.logits.ctx });
            so.logits = .{};
        }
        if (so.hidden.ctx != null) {
            if (p.f.want_hidden) try p.hidden.append(gpa, .{ .ctx = so.hidden.ctx }) else _ = mlx.mlx_array_free(.{ .ctx = so.hidden.ctx });
            so.hidden = .{};
        }
        if (so.captured) |c| for (c[0..p.f.capture_layers.len], 0..) |*x, i| {
            try p.captures[i].append(gpa, .{ .ctx = x.ctx });
            x.* = .{};
        };
    }

    /// A span a decode lane holds, as the chunk loop forwards it: its bank phase, the table's forward, the settle.
    fn laneSpan(p: *Pass, ids: Array, pos: u32, rows: u32, last: bool, why: *abi.Verdict) !void {
        var sub = p.f.*;
        const at = [_]u32{pos};
        sub.mode = if (rows == 1) .decode else .prefill;
        sub.rows = rows;
        sub.tokens = .{};
        sub.ids = .{ .ctx = ids.ctx };
        sub.positions = .of(&at);
        sub.logits = if (p.f.logits == .all_rows) .all_rows else if (last) p.f.logits else .none;
        sub.progress = null;
        sub.prompt_spans = .{};
        sub.ssm_checkpoints = .{};
        var captured: [max_captures]abi.Array = @splat(.{});
        var so: abi.ForwardOut = .{ .captured = &captured };
        try trunk.bankPhase(p.m, rows);
        var v = host.forwardOne(name, fwd, p.m, p.ops, &sub, &so);
        if (v.isOk()) v = host.settleBank(p.m, &sub, &so);
        if (!v.isOk()) {
            why.* = v;
            return error.PromptSpanRefused;
        }
        errdefer host.clear(&so, p.f.capture_layers.len);
        try p.take(&so);
        const sl = p.slots(&at);
        for (p.f.ssm_checkpoints.slice(), 0..) |cp, c| if (cp == @as(u64, pos) + rows) for (p.m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
            p.conv[c][li] = try fwd.kept(sl.state(0, @intCast(li), .conv), p.m.s);
            p.ssm[c][li] = try fwd.kept(sl.state(0, @intCast(li), .ssm), p.m.s);
        };
    }

    /// A run of wider spans in the bank's prompt phase, layer by layer (`hiddenSpans`), its head over the spans the
    /// forward reads logits of, evaluated, then settled. `done`: the prompt's rows before it.
    fn wideSegment(p: *Pass, ids: Array, pos: u32, rows: []const u32, last: bool, done: u64) !void {
        var seg_rows: u32 = 0;
        for (rows) |r| seg_rows += r;
        try trunk.bankPhase(p.m, seg_rows);
        var after: [max_checkpoints]usize = undefined;
        var conv: [max_checkpoints][]Array = undefined;
        var ssm: [max_checkpoints][]Array = undefined;
        var n_after: usize = 0;
        var end: u64 = pos;
        for (rows, 0..) |r, i| {
            end += r;
            for (p.f.ssm_checkpoints.slice(), 0..) |cp, c| if (cp == end) {
                if (n_after == after.len) return error.PromptTooManyCheckpoints;
                after[n_after] = i;
                conv[n_after] = p.conv[c];
                ssm[n_after] = p.ssm[c];
                n_after += 1;
            };
        }
        const n_cap = p.f.capture_layers.len;
        var cap_out: [max_captures]Array = undefined;
        for (cap_out[0..n_cap]) |*c| c.* = mlx.mlx_array_new();
        defer for (cap_out[0..n_cap]) |c| {
            _ = mlx.mlx_array_free(c);
        };
        const outs = try gpa.alloc(Array, rows.len);
        defer gpa.free(outs);
        @memset(outs, .{ .ctx = null });
        defer freeAll(outs);
        const at = [_]u32{pos};
        const sp: fwd.Spans = .{
            .slots = p.slots(&at),
            .rows = rows,
            .keep = .{ .after = after[0..n_after], .conv = conv[0..n_after], .ssm = ssm[0..n_after] },
            .capture_ids = p.f.capture_layers.slice(),
            .capture_out = cap_out[0..n_cap],
            .progress = p.f.progress,
            .done = done,
        };
        try fwd.hiddenSpans(p.m, &sp, ids, outs);
        // The head as the chunk loop's forward of each span runs it (`host.finish`), where the forward reads its logits.
        for (outs, 0..) |hd, i| {
            const want: abi.Logits = if (p.f.logits == .all_rows) .all_rows else if (last and i + 1 == outs.len) p.f.logits else .none;
            if (want == .none) continue;
            var sub = p.f.*;
            sub.logits = want;
            sub.want_hidden = false;
            var so: abi.ForwardOut = .{};
            var hr = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&hr, hd);
            try host.finish(p.m, &sub, hr, &so);
            errdefer host.clear(&so, 0);
            try p.take(&so);
        }
        if (p.f.want_hidden) for (outs) |hd| try p.hidden.append(gpa, refOf(hd));
        for (cap_out[0..n_cap], 0..) |c, i| try p.captures[i].append(gpa, refOf(c));
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (p.logits.items) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        for (p.hidden.items) |x| _ = mlx.mlx_vector_array_append_value(vec, x);
        for (p.captures[0..n_cap]) |c| for (c.items) |x| {
            _ = mlx.mlx_vector_array_append_value(vec, x);
        };
        try mlx.check(mlx.mlx_eval(vec));
        if (p.m.routed) |r| try r.settle();
    }
};

fn run(m: *trunk.Model, ops: *const abi.CacheOps, f: *const abi.Forward, out: *abi.ForwardOut, why: *abi.Verdict) !void {
    if (f.caches.len != 1 or f.positions.len != 1) return error.PromptOneSlot;
    if (f.media.ctx != null) return error.PromptMediaUnsupported;
    const n_cap = f.capture_layers.len;
    if (n_cap > max_captures) return error.TooManyCaptureLayers;
    if (n_cap > 0 and out.captured == null) return error.CaptureLayersWithoutOutput;
    const pos0: u64 = f.positions.slice()[0];
    const ends = f.prompt_spans.slice();
    const cps = f.ssm_checkpoints.slice();
    try span_plan.check(pos0, f.rows, ends, cps);
    const rows = try gpa.alloc(u32, ends.len);
    defer gpa.free(rows);
    var prev = pos0;
    for (ends, rows) |e, *r| {
        r.* = @intCast(e - prev);
        prev = e;
    }
    const n_layers = m.layers.len;
    const conv = try gpa.alloc([]Array, cps.len);
    defer gpa.free(conv);
    const ssm = try gpa.alloc([]Array, cps.len);
    defer gpa.free(ssm);
    var made: usize = 0;
    defer for (conv[0..made], ssm[0..made]) |cs, ss| {
        freeAll(cs);
        freeAll(ss);
        gpa.free(cs);
        gpa.free(ss);
    };
    while (made < cps.len) : (made += 1) {
        conv[made] = try gpa.alloc(Array, n_layers);
        ssm[made] = gpa.alloc(Array, n_layers) catch |e| {
            gpa.free(conv[made]);
            return e;
        };
        @memset(conv[made], .{ .ctx = null });
        @memset(ssm[made], .{ .ctx = null });
    }
    var p: Pass = .{ .m = m, .ops = ops, .f = f, .conv = conv, .ssm = ssm };
    defer p.deinit();

    const ids = try host.tokenIds(f);
    defer _ = mlx.mlx_array_free(ids);
    const top_k = m.c.num_experts_per_tok;
    const lane_ids: usize = if (m.routed) |r| r.decode_ids else 0;
    var j: usize = 0;
    var at: u32 = 0;
    while (j < rows.len) {
        const k = span_plan.segmentEnd(rows, j, top_k, lane_ids);
        var seg_rows: u32 = 0;
        for (rows[j..k]) |r| seg_rows += r;
        var idv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idv);
        try mlx.check(mlx.mlx_slice(&idv, ids, &[_]c_int{ 0, @intCast(at) }, 2, &[_]c_int{ 1, @intCast(at + seg_rows) }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
        const pos: u32 = @intCast(pos0 + at);
        if (span_plan.inLane(rows[j], top_k, lane_ids))
            try p.laneSpan(idv, pos, seg_rows, k == rows.len, why)
        else
            try p.wideSegment(idv, pos, rows[j..k], k == rows.len, at);
        at += seg_rows;
        j = k;
        if (f.progress) |pr| @atomicStore(u64, pr, at, .monotonic);
    }

    // The checkpoints, in order, once every layer has passed them.
    if (cps.len > 0) {
        const file = ops.checkpointOp() orelse return error.PromptCheckpointsNotServed;
        const states = try gpa.alloc(abi.LayerState, n_layers);
        defer gpa.free(states);
        for (cps, conv, ssm) |cp, cs, ss| {
            var ns: usize = 0;
            for (m.layers, 0..) |*lw, li| if (lw.mixer == .kda) {
                states[ns] = .{ .layer = @intCast(li), .conv = .{ .ctx = cs[li].ctx }, .ssm = .{ .ctx = ss[li].ctx } };
                ns += 1;
            };
            const v = file(f.caches.slice()[0], cp, .of(states[0..ns]));
            if (!v.isOk()) {
                why.* = v;
                return error.PromptCheckpointRefused;
            }
        }
    }

    if (p.logits.items.len > 0) out.logits = .{ .ctx = (try joined(&p.logits, 0, m.s)).ctx };
    if (f.want_hidden) out.hidden = .{ .ctx = (try joined(&p.hidden, 1, m.s)).ctx };
    for (p.captures[0..n_cap], 0..) |*c, i| out.captured.?[i] = .{ .ctx = (try joined(c, 1, m.s)).ctx };
}

/// `parts` joined along `axis` (a new reference; one part is moved out as it is).
fn joined(parts: *Parts, axis: c_int, s: mlx.mlx_stream) !Array {
    if (parts.items.len == 1) return parts.pop().?;
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_concatenate_axis(&r, vec, axis, s));
    return r;
}

fn refOf(x: Array) Array {
    var r = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&r, x);
    return r;
}

fn freeAll(xs: []Array) void {
    for (xs) |*x| if (x.ctx != null) {
        _ = mlx.mlx_array_free(x.*);
        x.* = .{ .ctx = null };
    };
}

fn freeParts(p: *Parts) void {
    for (p.items) |x| _ = mlx.mlx_array_free(x);
    p.deinit(gpa);
}
