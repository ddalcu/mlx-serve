//! The MTP head path of a served model proven against the legacy sidecar arm: a test head behind `SpecHooks.head`
//! runs the same qwen-layout head layer the legacy `MtpModel` runs (on the served trunk's config, as a plugin knows
//! its own), so what is under test is the engine's side: the `HeadHooks` call (ids, `hidden_in`, the head's cache
//! behind `cache_ops`, the row modes), the readout built from the shell's borrowed trunk head, and the generator's
//! rounds through `MtpHeadRef.plugin`. Head steps, draft picks and a pinned-depth round loop are bit-equal.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const fixture = @import("../../../sdk/bases/proof/fixture.zig");
const llama = @import("../../../plugins/llama/plugin.zig");
const proof = @import("proof.zig");
const plugin_head = @import("head.zig");
const cache_ops = @import("../kv/cache_ops.zig");
const depth_bounds = @import("../spec/mtp_depth_bounds.zig");
const dflash_mod = @import("../spec/dflash/dflash.zig");
const MtpHeadRef = @import("../spec/mtp_head_ref.zig").MtpHeadRef;
const MtpCacheRef = @import("../spec/mtp_head_ref.zig").MtpCacheRef;
const Generator = @import("../generate/generator.zig").Generator;
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const Transformer = @import("../transformer.zig").Transformer;

const testing = std.testing;
const Array = mlx.mlx_array;

/// The test head's layer and the trunk whose config it computes on.
var double: struct { m: mtp_mod.MtpModel = undefined, target: *Transformer = undefined } = .{};

fn hLoad(_: *anyopaque, _: *const abi.LoadCtx) callconv(.c) abi.Verdict {
    return .ok;
}
fn hSlot(_: *anyopaque, _: u32) callconv(.c) abi.Verdict {
    return .ok;
}
fn hRelease(_: *anyopaque, _: u32) callconv(.c) void {}
fn hTruncate(_: *anyopaque, _: u32, _: u64) callconv(.c) void {}
fn hAdopt(_: *anyopaque, _: u32, _: abi.Handle) callconv(.c) abi.Verdict {
    return .ok;
}
fn hSelect(_: *anyopaque, _: abi.Array, _: *abi.Array) callconv(.c) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "the engine selects drafts");
}
fn hBill(_: *const abi.Parsed) callconv(.c) u64 {
    return 0;
}

/// The legacy sidecar's three row modes, as a plugin head maps them.
fn hForward(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    headForward(f, out) catch return abi.Verdict.refuse(.failed, "test head forward");
    return .ok;
}

fn headForward(f: *const abi.Forward, out: *abi.ForwardOut) !void {
    const slot: *cache_ops.Slot = @ptrCast(@alignCast(f.caches.slice()[0]));
    const s = double.target.s;
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_reshape(&ids, .{ .ctx = f.ids.ctx }, &[_]c_int{@intCast(f.rows)}, 1, s));
    const hidden: Array = .{ .ctx = f.hidden_in.ctx };
    const off: c_int = @intCast(f.positions.slice()[0]);
    if (f.logits == .none and !f.want_hidden) return mtp_mod.appendKvOnly(&double.m, double.target, slot.kv, ids, hidden, off, null);
    const r = try mtp_mod.forwardWithMrope(&double.m, double.target, slot.kv, ids, hidden, off, f.logits == .last_row, null);
    if (r.logits.ctx != null) _ = mlx.mlx_array_free(r.logits);
    out.hidden = .{ .ctx = r.hidden_next.ctx };
}

const head_hooks: abi.HeadHooks = .{
    .load = &hLoad,
    .activate = &hSlot,
    .release = &hRelease,
    .forward = &hForward,
    .truncate = &hTruncate,
    .adopt = &hAdopt,
    .mark = &hRelease,
    .draft_select = &hSelect,
    .bill = &hBill,
};
const spec_hooks: abi.SpecHooks = .{ .head = &head_hooks, .trunk_weight = llama.table.spec.?.trunk_weight };
const test_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &spec_hooks;
    break :blk t;
};

/// A head whose files are not there: an MLX read fails (latching its error) before the load refuses.
fn hLoadFails(_: *anyopaque, _: *const abi.LoadCtx) callconv(.c) abi.Verdict {
    const v = [_]f32{ 1, 2, 3 };
    const a = mlx.mlx_array_new_data(&v, &[_]c_int{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(a);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    _ = mlx.mlx_reshape(&x, a, &[_]c_int{2}, 1, fixture.stream());
    return abi.Verdict.refuse(.failed, "no head");
}
const failing_head_hooks: abi.HeadHooks = blk: {
    var h = head_hooks;
    h.load = &hLoadFails;
    break :blk h;
};
const failing_spec_hooks: abi.SpecHooks = .{ .head = &failing_head_hooks, .trunk_weight = llama.table.spec.?.trunk_weight };
const failing_table: abi.ModelTable = blk: {
    var t = llama.table;
    t.spec = &failing_spec_hooks;
    break :blk t;
};

/// A qwen-layout dense head (`mtp.*`, the q projection carrying the output gate) for the proof trunk's geometry.
fn writeHead(path: []const u8, g: proof.Geo) !void {
    const s = fixture.stream();
    const Fx = dflash_mod.TinyFix;
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    const h = g.hidden;
    try Fx.putW(map, "mtp.fc.weight", h, 2 * h, 201, s);
    for ([_][]const u8{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight" }) |k|
        try Fx.putNorm(map, k, h, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.q_norm.weight", g.head_dim, s);
    try Fx.putNorm(map, "mtp.layers.0.self_attn.k_norm.weight", g.head_dim, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.q_proj.weight", 2 * g.heads * g.head_dim, h, 202, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.k_proj.weight", g.kv_heads * g.head_dim, h, 203, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.v_proj.weight", g.kv_heads * g.head_dim, h, 204, s);
    try Fx.putW(map, "mtp.layers.0.self_attn.o_proj.weight", h, g.heads * g.head_dim, 205, s);
    try Fx.putW(map, "mtp.layers.0.mlp.gate_proj.weight", g.inter, h, 206, s);
    try Fx.putW(map, "mtp.layers.0.mlp.up_proj.weight", g.inter, h, 207, s);
    try Fx.putW(map, "mtp.layers.0.mlp.down_proj.weight", h, g.inter, 208, s);
    const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model-mtp.safetensors", .{path}, 0);
    defer testing.allocator.free(file);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
}

fn hiddenRows(g: *ops.Graph, rows: u32, hidden: u32, seed: u64) !Array {
    var prng = std.Random.DefaultPrng.init(seed);
    const buf = try testing.allocator.alloc(f32, rows * hidden);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32) * 0.5;
    const f = try g.adopt(mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(hidden) }, 3, .float32));
    return g.astype(f, .bfloat16);
}

fn idArr(g: *ops.Graph, ids: []const i32) !Array {
    return g.adopt(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .int32));
}

/// A greedy pinned-depth MTP run: the emitted stream and each round's accepted count.
pub fn mtpRun(xfm: *Transformer, head: MtpHeadRef, tokens: []u32, accepted: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try xfm.resetCache();
    var tok_dummy: Tokenizer = undefined; // never read by the Generator
    const prompt = [_]u32{ 3, 7, 1, 12, 30, 5, 9, 22, 4, 17, 2, 28 };
    var gen = try Generator.initWithOptions(io, a, xfm, &tok_dummy, &prompt, @intCast(tokens.len), .{ .temperature = 0.0 }, &.{}, .{
        .mtp_enabled = true,
        .mtp = head,
        .model_has_mtp = true,
        .mtp_depth = 2,
    });
    defer gen.deinit(a);
    var n: usize = 0;
    while (try gen.nextMtp(a)) |res| {
        defer a.free(res.tokens);
        @memcpy(tokens[n..][0..res.tokens.len], res.tokens);
        n += res.tokens.len;
        try accepted.append(a, res.accepted_tokens);
    }
    try testing.expectEqual(tokens.len, n);
    try testing.expect(gen.mtp_attempted > 0);
}

// ── tests ──

test "plugin MTP head: a head that fails to load leaves no MLX error for the first request" {
    mlx.installErrorHandler();
    const io = std.Io.Threaded.global_single_threaded.io();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x3e7b);
    defer b.close();
    b.xfm.plugin.?.table = &failing_table;
    try testing.expect(!mlx.errorPending());
    try testing.expect(plugin_head.load(io, testing.allocator, &b.xfm, b.dir) == null);
    try testing.expect(!mlx.errorPending());
}

test "plugin MTP head: head steps, draft picks and a pinned-depth round loop equal the legacy sidecar arm" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const geo: proof.Geo = .{ .quantized = true };
    var b: proof.Served = undefined;
    try b.open(geo, 0x3e7a);
    defer b.close();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const head_dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try writeHead(head_dir, geo);
    const s = fixture.stream();

    // Legacy: the sidecar bound to the served trunk (the qwen head's output gate, which a llama trunk never reads).
    b.xfm.config.attn_output_gate = true;
    var legacy = try mtp_mod.loadMtp(io, a, s, head_dir);
    defer legacy.deinit();
    try legacy.bind(&b.xfm);
    // Served: the test table's head over the same shell.
    double.m = try mtp_mod.loadMtp(io, a, s, head_dir);
    defer double.m.deinit();
    double.target = &b.xfm;
    b.xfm.plugin.?.table = &test_table;
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    try testing.expectEqual(legacy.draft_head != null, served.draft_head != null);
    try testing.expectEqual(legacy.rerank_coarse != null, served.rerank_coarse != null);

    const heads = [_]MtpHeadRef{ .{ .qwen = &legacy }, .{ .plugin = served } };
    const xs = [_]*Transformer{ &b.xfm, &b.xfm };
    var caches: [2]MtpCacheRef = undefined;
    for (heads, &caches) |h, *c| c.* = try h.makeCache(a);
    defer for (&caches) |*c| c.deinit();
    {
        var g = ops.Graph.init(a, s);
        defer g.deinit();
        const hist = try hiddenRows(&g, 3, geo.hidden, 1);
        const step = try hiddenRows(&g, 2, geo.hidden, 2);
        const one = try hiddenRows(&g, 1, geo.hidden, 3);
        var outs: [2][4]Array = undefined;
        for (heads, xs, &caches, 0..) |h, x, *c, k| {
            try h.appendHistory(x, c, &.{ 5, 9, 40 }, hist, 0, null, a);
            const merged = try h.forward(x, c, try idArr(&g, &.{ 7, 11 }), step, 3, .logits, null, null);
            const mixed = try h.forward(x, c, try idArr(&g, &.{60}), one, 5, .mixed, null, null);
            outs[k] = .{ try g.adopt(merged.logits), try g.adopt(merged.hidden_next), try g.adopt(mixed.hidden_next), try g.adopt(try h.draftSelect(x, mixed.hidden_next, null)) };
            try testing.expectEqual(@as(usize, 6), c.step());
        }
        for (outs[0], outs[1]) |want, got| try ops.expectBitEqual(got, want);
    }

    depth_bounds.active = .{ .min = 2, .max = 2 };
    defer depth_bounds.active = .{};
    var tokens: [2][16]u32 = undefined;
    var accepted: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&accepted) |*l| l.deinit(a);
    for (heads, xs, 0..) |h, x, k| try mtpRun(x, h, &tokens[k], &accepted[k]);
    try testing.expectEqualSlices(u32, &tokens[0], &tokens[1]);
    try testing.expectEqualSlices(u32, accepted[0].items, accepted[1].items);
}

/// A three-head double (MiMo's shape): records which head each forward ran and hands the trunk hidden back.
const Heads3 = struct {
    var ran: [8]u32 = undefined;
    var n: usize = 0;
    var histories: u32 = 0;

    fn forward(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
        if (f.logits == .none and !f.want_hidden) {
            histories += 1;
            return .ok;
        }
        ran[n] = f.head;
        n += 1;
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, .{ .ctx = f.hidden_in.ctx });
        out.hidden = .{ .ctx = h.ctx };
        return .ok;
    }
    const hooks: abi.HeadHooks = blk: {
        var hk = head_hooks;
        hk.forward = &forward;
        break :blk hk;
    };
    const spec: abi.SpecHooks = .{ .head = &hooks, .trunk_weight = llama.table.spec.?.trunk_weight };
    const table: abi.ModelTable = blk: {
        var t = llama.table;
        t.spec = &spec;
        break :blk t;
    };
};

test "plugin MTP heads: a round opens at head 0, each chained draft runs the next head, the trunk hidden carries on" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const geo: proof.Geo = .{ .quantized = true };
    var b: proof.Served = undefined;
    try b.open(geo, 0x3e7b);
    defer b.close();
    b.xfm.plugin.?.table = &Heads3.table;
    b.xfm.config.plugin.?.parsed.spec.mtp_heads = 3;
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    const h: MtpHeadRef = .{ .plugin = served };
    try testing.expectEqual(@as(u32, 3), h.maxDepth());
    var c = try h.makeCache(a);
    defer c.deinit();
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    const trunk = try hiddenRows(&g, 1, geo.hidden, 4);
    Heads3.n = 0;
    Heads3.histories = 0;
    try h.appendHistory(&b.xfm, &c, &.{ 5, 9 }, try hiddenRows(&g, 2, geo.hidden, 5), 0, null, a);
    // A step that wants nothing appends history (MiMo's legacy arm), an opening runs head 0.
    _ = try g.adopt((try h.forward(&b.xfm, &c, try idArr(&g, &.{7}), trunk, 2, .none, null, &.{7})).hidden_next);
    try testing.expectEqual(@as(u32, 2), Heads3.histories);
    try testing.expectEqual(@as(usize, 0), Heads3.n);
    const o = try h.forward(&b.xfm, &c, try idArr(&g, &.{7}), trunk, 2, .mixed, null, &.{7});
    _ = try g.adopt(o.rerank_x);
    _ = try g.adopt(o.hidden_next);
    var hidden = trunk;
    for (0..3) |_| {
        const r = try h.forward(&b.xfm, &c, try idArr(&g, &.{11}), hidden, 3, .logits, null, null);
        _ = try g.adopt(r.logits);
        hidden = try g.adopt(r.hidden_next);
        try ops.expectBitEqual(hidden, trunk);
    }
    try testing.expectEqual(@as(u32, 2), Heads3.histories);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 2, 2 }, Heads3.ran[0..Heads3.n]);
    try c.truncate(2, fixture.stream());
    _ = try h.forward(&b.xfm, &c, try idArr(&g, &.{11}), trunk, 3, .mixed, null, null);
    try testing.expectEqual(@as(u32, 0), Heads3.ran[Heads3.n - 1]);
}

/// A head that records each history append's rows and position and hands the trunk hidden back otherwise.
const History = struct {
    var rows: [16]u32 = undefined;
    var at: [16]u32 = undefined;
    var n: usize = 0;

    fn forward(_: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
        if (f.logits == .none and !f.want_hidden) {
            if (n < rows.len) {
                rows[n] = f.rows;
                at[n] = f.positions.slice()[0];
            }
            n += 1;
            return .ok;
        }
        var h = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&h, .{ .ctx = f.hidden_in.ctx });
        out.hidden = .{ .ctx = h.ctx };
        return .ok;
    }
    const hooks: abi.HeadHooks = blk: {
        var hk = head_hooks;
        hk.forward = &forward;
        break :blk hk;
    };
    const spec: abi.SpecHooks = .{ .head = &hooks, .trunk_weight = llama.table.spec.?.trunk_weight };
    const table: abi.ModelTable = blk: {
        var t = llama.table;
        t.spec = &spec;
        break :blk t;
    };
};

// Bar: a whole-prompt prefill reserves the chunk loop's KV capacity and appends its MTP history in the chunk loop's
// appends (rows and positions), never one.
test "plugin MTP head: a whole-prompt prefill reserves the chunk loop's KV and appends the history over its spans" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var b: proof.Served = undefined;
    try b.open(.{ .quantized = true }, 0x3e7c);
    defer b.close();
    b.xfm.plugin.?.table = &History.table;
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    // 64-token chunks with the 512-token tail merge: the 507 tokens past 192 join the third chunk, so the spans end at 64,
    // 128 and 699, then the last token.
    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(4 + (i * 37) % 250);
    var got: [2][]const u32 = undefined;
    var got_at: [2][]const u32 = undefined;
    var bufs: [2][2][16]u32 = undefined;
    for ([_]bool{ true, false }, 0..) |whole, k| {
        b.xfm.config.plugin.?.parsed.caps.whole_prompt_prefill = whole;
        try b.xfm.resetCache();
        History.n = 0;
        var tok: Tokenizer = undefined; // never read by the Generator
        var gen = try Generator.initWithOptions(io, a, &b.xfm, &tok, &prompt, 2, .{ .temperature = 0.0 }, &.{}, .{
            .mtp_enabled = true,
            .mtp = .{ .plugin = served },
            .model_has_mtp = true,
            .mtp_depth = 1,
            .pinned_prefill_chunk = 64,
        });
        defer gen.deinit(a);
        // The fixture's cache starts unreserved: 700 prompt tokens plus one 64-token chunk of slack.
        if (whole) try testing.expectEqual(@as(usize, 764), gen.ctx.cache.reserve_tokens);
        try testing.expect(History.n <= History.rows.len);
        bufs[k][0] = History.rows;
        bufs[k][1] = History.at;
        got[k] = bufs[k][0][0..History.n];
        got_at[k] = bufs[k][1][0..History.n];
    }
    try testing.expectEqualSlices(u32, &.{ 64, 64, 571 }, got[1]);
    try testing.expectEqualSlices(u32, got[0], got[1]);
    try testing.expectEqualSlices(u32, got_at[0], got_at[1]);
}

/// A whole-prompt model that chunks its own prompt (`Caps.whole_prompt_prefill`, as deepseek_v41 runs its own calls):
/// handed the chunk loop's spans (`Forward.prompt_spans`), it runs llama's forward a span, each span's logits and
/// hidden rows joined, so its rows are the chunk loop's; any other forward is llama's. Its head is the test head.
const SelfChunked = struct {
    fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
        if (f.mode != .prefill or f.prompt_spans.len == 0) return llama.table.forward(mp, f, out);
        spans(mp, f, out) catch return abi.Verdict.refuse(.failed, "self-chunked prompt");
        return .ok;
    }

    fn spans(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) !void {
        if (f.ids.ctx == null or f.capture_layers.len > 0) return error.SelfChunkedForward;
        var g = ops.Graph.init(testing.allocator, fixture.stream());
        defer g.deinit();
        var logits: std.ArrayList(Array) = .empty;
        defer logits.deinit(testing.allocator);
        var hidden: std.ArrayList(Array) = .empty;
        defer hidden.deinit(testing.allocator);
        const ids: Array = .{ .ctx = f.ids.ctx };
        const pos0: u64 = f.positions.slice()[0];
        const ends = f.prompt_spans.slice();
        var lo = pos0;
        for (ends, 0..) |hi, k| {
            const at = [_]u32{@intCast(lo)};
            var sub = f.*;
            sub.mode = if (hi - lo == 1) .decode else .prefill;
            sub.rows = @intCast(hi - lo);
            sub.positions = .of(&at);
            sub.ids = .{ .ctx = (try g.sliceAxis(ids, 1, @intCast(lo - pos0), @intCast(hi - pos0))).ctx };
            sub.logits = if (f.logits == .last_row and k + 1 < ends.len) .none else f.logits;
            sub.prompt_spans = .{};
            sub.ssm_checkpoints = .{};
            sub.progress = null;
            var so: abi.ForwardOut = .{};
            const v = llama.table.forward(mp, &sub, &so);
            if (so.logits.ctx != null) try logits.append(testing.allocator, try g.adopt(.{ .ctx = so.logits.ctx }));
            if (so.hidden.ctx != null) try hidden.append(testing.allocator, try g.adopt(.{ .ctx = so.hidden.ctx }));
            if (!v.isOk()) return error.SelfChunkedSpan;
            lo = hi;
        }
        if (logits.items.len > 0) out.logits = .{ .ctx = g.escape(try g.concat(logits.items, 0)).ctx };
        if (hidden.items.len > 0) out.hidden = .{ .ctx = g.escape(try g.concat(hidden.items, 1)).ctx };
    }

    const table: abi.ModelTable = blk: {
        var t = llama.table;
        t.spec = &spec_hooks;
        t.forward = &forward;
        break :blk t;
    };
};

// Bar: a model that chunks its own whole prompt reserves the chunk loop's KV capacity and leaves the MTP head's history
// (rows, keys and values) the chunk loop's, bit for bit.
test "plugin MTP head: a self-chunking whole-prompt model's KV reservation and MTP history are the chunk loop's, bit for bit" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const geo: proof.Geo = .{ .quantized = true };
    var b: proof.Served = undefined;
    try b.open(geo, 0x3e7d);
    defer b.close();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const head_dir = buf[0..try tmp.dir.realPath(io, &buf)];
    try writeHead(head_dir, geo);
    // The qwen-layout head's q projection carries its output gate (a llama trunk never reads the flag).
    b.xfm.config.attn_output_gate = true;
    double.m = try mtp_mod.loadMtp(io, a, fixture.stream(), head_dir);
    defer double.m.deinit();
    double.target = &b.xfm;
    b.xfm.plugin.?.table = &SelfChunked.table;
    const served = plugin_head.load(io, a, &b.xfm, b.dir) orelse return error.TestHeadNotLoaded;
    defer plugin_head.destroy(served, a);
    // 64-token chunks with the 512-token tail merge: the 507 tokens past 192 join the third chunk, so the spans end at 64,
    // 128 and 699, then the last token.
    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(4 + (i * 37) % 250);
    var g = ops.Graph.init(a, fixture.stream());
    defer g.deinit();
    var reserved: [2]usize = undefined;
    var rows: [2]usize = undefined;
    var keys: [2]Array = undefined;
    var values: [2]Array = undefined;
    for ([_]bool{ false, true }, 0..) |whole, k| {
        b.xfm.config.plugin.?.parsed.caps.whole_prompt_prefill = whole;
        try b.xfm.resetCache();
        var tok: Tokenizer = undefined; // never read by the Generator
        var gen = try Generator.initWithOptions(io, a, &b.xfm, &tok, &prompt, 2, .{ .temperature = 0.0 }, &.{}, .{
            .mtp_enabled = true,
            .mtp = .{ .plugin = served },
            .model_has_mtp = true,
            .mtp_depth = 1,
            .pinned_prefill_chunk = 64,
        });
        defer gen.deinit(a);
        reserved[k] = gen.ctx.cache.reserve_tokens;
        const e = gen.mtp_cache.?.plugin.kv.entries[0];
        rows[k] = e.offset - e.base;
        keys[k] = try g.sliceAxis(e.keys, 2, 0, @intCast(rows[k]));
        values[k] = try g.sliceAxis(e.values, 2, 0, @intCast(rows[k]));
        try ops.eval(keys[k]);
        try ops.eval(values[k]);
    }
    // The chunk loop's reservation (`reservedPrefillTokens`: 700 prompt tokens and a 64-token chunk) on both paths.
    try testing.expectEqual(@as(usize, 764), reserved[0]);
    try testing.expectEqual(reserved[0], reserved[1]);
    // The history holds the prompt's 699 pairs (the last row's pair is the first round's), the same words on both.
    try testing.expectEqual(@as(usize, 699), rows[0]);
    try testing.expectEqual(rows[0], rows[1]);
    try ops.expectBitEqual(keys[1], keys[0]);
    try ops.expectBitEqual(values[1], values[0]);
}
