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

const history_env = @import("../../../env.zig");
const token_history = @import("../spec/token_history.zig");
const history_trace = @import("../spec/mtp/trace.zig");
const history_verify = @import("../spec/verify.zig");
const HistoryKV = @import("../kv/cache.zig").KVCache;
const HistorySnapshot = @import("../kv/cache.zig").KVCacheSnapshot;

/// Real served trunk/head and a parsed, tokenizer-bound vocabulary. The tokenizer
/// parser owns its strings and digest after the temporary JSON buffer disappears.
const HistoryRig = struct {
    b: proof.Served,
    head: *plugin_head.Head,
    tok: Tokenizer,

    fn open(self: *HistoryRig) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const geo: proof.Geo = .{ .quantized = true };
        try self.b.open(geo, 0x3e7c);
        errdefer self.b.close();
        try writeHead(self.b.dir, geo);
        self.b.xfm.config.attn_output_gate = true;
        double.m = try mtp_mod.loadMtp(io, a, fixture.stream(), self.b.dir);
        errdefer double.m.deinit();
        double.target = &self.b.xfm;
        self.b.xfm.plugin.?.table = &test_table;
        self.head = plugin_head.load(io, a, &self.b.xfm, self.b.dir) orelse return error.TestHeadNotLoaded;
        errdefer plugin_head.destroy(self.head, a);
        var buf: [8192]u8 = undefined;
        var json: std.Io.Writer = .fixed(&buf);
        try json.writeAll("{\"model\":{\"type\":\"BPE\",\"vocab\":{");
        for (0..geo.vocab) |i| try json.print("{s}\"t{d}\":{d}", .{ if (i == 0) "" else ",", i, i });
        try json.writeAll("},\"merges\":[]}}");
        self.tok = try @import("../../../sdk/tokenizer/load.zig").parseTokenizerContent(io, a, json.buffered());
    }

    fn close(self: *HistoryRig) void {
        self.tok.deinit();
        plugin_head.destroy(self.head, testing.allocator);
        double.m.deinit();
        self.b.close();
    }

    fn generator(self: *HistoryRig, prompt: []const u32) !Generator {
        try self.b.xfm.resetCache();
        return Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), testing.allocator, &self.b.xfm, &self.tok, prompt, 64, .{ .temperature = 0 }, &.{}, .{
            .mtp_enabled = true,
            .mtp = .{ .plugin = self.head },
            .model_has_mtp = true,
            .mtp_depth = 3,
            .skip_lazy_preforward = true,
        });
    }

    fn store(self: *HistoryRig, name: []const u8, prompt: []const u32, continuation: []const u32) ![:0]u8 {
        const a = testing.allocator;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ self.b.dir, name }, 0);
        errdefer a.free(path);
        var cache = token_history.HistoryCache.init(a);
        defer cache.deinit() catch unreachable;
        try cache.appendFinalTurn(std.Io.Threaded.global_single_threaded.io(), path, self.tok.content_sha256.?, prompt, continuation);
        try cache.appendFinalTurn(std.Io.Threaded.global_single_threaded.io(), path, self.tok.content_sha256.?, &history_other_query, &history_native);
        return path;
    }
};

const history_query = [_]u32{ 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 };
const history_external = [_]u32{ 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173 };
const history_native = [_]u32{ 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193 };
const history_other_query = [_]u32{ 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95 };

test "plugin MTP history: native incumbent wins conflicts; rejected native admits owned external chain" {
    const a = testing.allocator;
    var corpus = history_env.override(.MLX_SERVE_LOOKUP_CORPUS, null);
    defer corpus.restore();
    var readonly = history_env.override(.MLX_SERVE_LOOKUP_READ_ONLY, "1");
    defer readonly.restore();
    const bounds = depth_bounds.active;
    depth_bounds.active = .{};
    defer depth_bounds.active = bounds;
    var rig: HistoryRig = undefined;
    try rig.open();
    defer rig.close();
    const path = try rig.store("incumbent", &history_query, &history_external);
    defer a.free(path);
    var store = history_env.override(.MLX_SERVE_LOOKUP_STORE, path.ptr);
    defer store.restore();

    for ([_]usize{ 16, 7 }) |native_order| {
        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(a);
        try prompt.appendSlice(a, history_query[16 - native_order ..]);
        try prompt.appendSlice(a, &history_native);
        try prompt.append(a, 220);
        try prompt.appendSlice(a, history_query[0..15]);
        var gen = try rig.generator(prompt.items);
        defer gen.deinit(a);
        gen.mtp_lookup_streak = true;
        gen.mtp_round_ema = 0;
        const plan = gen.mtpRoundPlan();
        var chain = (try gen.mtpLookupChain(a, plan, 47)) orelse return error.ExpectedHistoryChain;
        defer chain.deinit(a);
        try testing.expect(chain.lookup);
        try testing.expectEqual(@as(u32, 14), chain.m);
        try testing.expectEqual(native_order == 7, chain.external_history);
        try testing.expectEqualSlices(u32, if (native_order == 16) &history_native else &history_external, chain.drafts);
        if (native_order == 16) {
            // Admission of the incumbent must not even allocate an external request.
            try testing.expect(gen.external_history == null);
        } else {
            const request = gen.external_history.?;
            try testing.expectEqual(@as(u32, 16), request.matched_order);
            // Change both the candidate and cap in the request's reusable buffer.
            _ = try request.match(history_query[0..15], 47, 1);
            const replacement = (try request.match(history_other_query[0..15], 95, 14)) orelse return error.ExpectedReplacementCandidate;
            try testing.expectEqualSlices(u32, &history_native, replacement.draft);
            try testing.expectEqualSlices(u32, &history_external, chain.drafts);
            for (chain.draft_arrs, history_external) |arr, want| {
                try mlx.check(mlx.mlx_array_eval(arr));
                var got: i32 = -1;
                try mlx.check(mlx.mlx_array_item_int32(&got, arr));
                try testing.expectEqual(@as(i32, @intCast(want)), got);
            }
        }
    }
}

test "plugin MTP history: durable suffix boundaries obey the stock gate depth and remaining budget" {
    const a = testing.allocator;
    var corpus = history_env.override(.MLX_SERVE_LOOKUP_CORPUS, null);
    defer corpus.restore();
    var readonly = history_env.override(.MLX_SERVE_LOOKUP_READ_ONLY, "1");
    defer readonly.restore();
    const bounds = depth_bounds.active;
    defer depth_bounds.active = bounds;
    var rig: HistoryRig = undefined;
    try rig.open();
    defer rig.close();
    for ([_]u32{ 7, 8, 15, 16 }) |order| {
        depth_bounds.active = .{};
        var name: [32]u8 = undefined;
        const path = try rig.store(try std.fmt.bufPrint(&name, "suffix-{d}", .{order}), history_query[16 - order ..], &history_external);
        defer a.free(path);
        var store = history_env.override(.MLX_SERVE_LOOKUP_STORE, path.ptr);
        defer store.restore();
        var gen = try rig.generator(history_query[0..15]);
        defer gen.deinit(a);
        gen.mtp_lookup_streak = true;
        gen.mtp_round_ema = 0;
        const plan = gen.mtpRoundPlan();
        const got = try gen.mtpLookupChain(a, plan, 47);
        if (order == 7) {
            try testing.expect(got == null);
            try testing.expectEqual(@as(u32, 7), gen.external_history.?.matched_order);
            continue;
        }
        var chain = got orelse return error.ExpectedHistoryChain;
        defer chain.deinit(a);
        const want: u32 = if (order == 16) 14 else 7;
        try testing.expect(chain.external_history);
        try testing.expectEqual(want, chain.m);
        try testing.expectEqualSlices(u32, history_external[0..want], chain.drafts);

        depth_bounds.active = .{ .max = 5 };
        var capped = (try gen.mtpLookupChain(a, plan, 47)) orelse return error.ExpectedDepthClippedChain;
        defer capped.deinit(a);
        try testing.expectEqual(@as(u32, 5), capped.m);
        gen.max_tokens = 4;
        var budgeted = (try gen.mtpLookupChain(a, plan, 47)) orelse return error.ExpectedBudgetClippedChain;
        defer budgeted.deinit(a);
        try testing.expectEqual(@as(u32, 3), budgeted.m);
        depth_bounds.active = .{ .min = 4, .max = 5 };
        try testing.expect(try gen.mtpLookupChain(a, plan, 47) == null);
        gen.max_tokens = 1;
        try testing.expectEqual(@as(u32, 0), try gen.mtpLookupDrafts(a, plan, 47));
        gen.max_tokens = 64;
        depth_bounds.active = .{};
        gen.mtp_lookup_streak = false;
        gen.mtp_lookup_ema = 0;
        gen.mtp_round_ema = 14;
        // A cached external candidate must be readmitted, not cache the old gate.
        try testing.expect(try gen.mtpLookupChain(a, plan, 47) == null);
    }
}

/// A forward from a cloned boundary, not a serial-decode oracle. Full verify
/// comparisons use the identical row geometry; rollback comparisons use the
/// committed-prefix geometry the dense trunk's restore/re-forward arm uses.
const HistoryReplay = struct {
    cache: HistoryKV,
    logits: Array,
    hidden: Array,
    all: Array,

    fn run(xfm: *Transformer, snapshot: *const HistorySnapshot, offset: usize, ids: []const u32, verify: bool) !HistoryReplay {
        var self: HistoryReplay = .{
            .cache = try HistoryKV.init(testing.allocator, xfm.config.num_hidden_layers),
            .logits = .{ .ctx = null },
            .hidden = mlx.mlx_array_new(),
            .all = mlx.mlx_array_new(),
        };
        errdefer self.deinit();
        try self.cache.restore(snapshot);
        var off = offset;
        var ctx: @import("../forward_ctx.zig").ForwardCtx = .{
            .cache = &self.cache,
            .moe_seq_offset = &off,
            .ssm_entries = null,
            .capture_hidden = null,
            .vision_embeddings = null,
            .spec_verify = verify,
        };
        const signed = try testing.allocator.alloc(i32, ids.len);
        defer testing.allocator.free(signed);
        for (signed, ids) |*dst, src| dst.* = @intCast(src);
        const input = mlx.mlx_array_new_data(signed.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        self.logits = try xfm.forwardWithCaptureAll(&ctx, input, &self.hidden, &self.all);
        return self;
    }

    fn deinit(self: *HistoryReplay) void {
        self.cache.deinit();
        if (self.logits.ctx != null) _ = mlx.mlx_array_free(self.logits);
        _ = mlx.mlx_array_free(self.hidden);
        _ = mlx.mlx_array_free(self.all);
    }
};

fn expectHistoryKV(got: *const HistoryKV, want: *const HistoryKV) !void {
    try testing.expectEqual(want.step, got.step);
    try testing.expectEqual(want.entries.len, got.entries.len);
    for (got.entries, want.entries) |g, w| {
        try testing.expectEqual(w.offset, g.offset);
        try testing.expectEqual(w.initialized, g.initialized);
        if (w.initialized) {
            try ops.expectBitEqual(g.key_view, w.key_view);
            try ops.expectBitEqual(g.value_view, w.value_view);
        }
    }
}

fn historyBegin(gen: *Generator) !Generator.MtpRoundState {
    return switch (try gen.mtpRoundBegin(testing.allocator)) {
        .verify => |st| st,
        .open => |o| try gen.mtpRoundContinue(testing.allocator, o),
        .done => |result| {
            if (result) |r| testing.allocator.free(r.tokens);
            return error.ExpectedHistoryRound;
        },
    };
}

fn historyFinish(gen: *Generator, st: *Generator.MtpRoundState, expected: ?u32) !u32 {
    const a = testing.allocator;
    const s = gen.xfm.s;
    const ids = try a.alloc(u32, st.verify_len);
    defer a.free(ids);
    ids[0] = st.chain.t1;
    for (st.chain.draft_arrs[0..st.chain.m], ids[1..]) |arr, *id| {
        try mlx.check(mlx.mlx_array_eval(arr));
        var value: i32 = -1;
        try mlx.check(mlx.mlx_array_item_int32(&value, arr));
        id.* = @intCast(value);
    }
    var snapshot = try gen.ctx.cache.snapshot();
    defer snapshot.deinit();
    var reference = try HistoryReplay.run(gen.xfm, &snapshot, gen.ctx.moe_seq_offset.*, ids, true);
    defer reference.deinit();
    var am = try history_verify.verifyArgmax(reference.logits, gen.sampling.suppress_mask, s);
    defer am.deinit();
    try mlx.check(mlx.mlx_array_eval(am.lazy()));
    const predictions = try am.ids(ids.len);
    var accepted: u32 = 0;
    while (accepted < st.chain.m and ids[1 + accepted] == @as(u32, @intCast(predictions[accepted]))) : (accepted += 1) {}
    if (expected) |want| try testing.expectEqual(want, accepted);
    const next: u32 = @intCast(predictions[accepted]);
    const kept = 1 + accepted;

    var committed: ?HistoryReplay = null;
    defer if (committed) |*r| r.deinit();
    if (accepted < st.chain.m) {
        // This fixture is an attention-only restore/re-forward trunk.
        try testing.expect(!gen.xfm.config.caps().verify_truncates);
        committed = try HistoryReplay.run(gen.xfm, &snapshot, gen.ctx.moe_seq_offset.*, ids[0..kept], false);
    }
    const want_cache = if (committed) |*r| &r.cache else &reference.cache;
    const want_hidden = if (committed) |r| r.hidden else reference.hidden;

    var g = ops.Graph.init(a, s);
    defer g.deinit();
    var history_hidden = gen.last_hidden;
    if (accepted > 0) {
        var prefix = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&prefix, reference.all, &[_]c_int{ 0, 0, 0 }, 3,
            &[_]c_int{ 1, @intCast(accepted), mlx.getShape(reference.all)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        _ = try g.adopt(prefix);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, gen.last_hidden);
        _ = mlx.mlx_vector_array_append_value(vec, prefix);
        var joined = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
        history_hidden = try g.adopt(joined);
    }
    // Pin the pre-round hidden across the finish, which replaces gen.last_hidden.
    var held_hidden = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&held_hidden, history_hidden));
    _ = try g.adopt(held_hidden);
    var head_snapshot = try gen.mtp_cache.?.plugin.kv.snapshot();
    defer head_snapshot.deinit();
    var head_reference = try gen.mtp.?.makeCache(a);
    defer head_reference.deinit();
    try head_reference.plugin.kv.restore(&head_snapshot);
    try head_reference.truncate(st.chain.off0, s);
    const signed = try a.alloc(i32, kept);
    defer a.free(signed);
    for (signed, ids[0..kept]) |*dst, src| dst.* = @intCast(src);
    const head_ids = try idArr(&g, signed);
    const head_out = try gen.mtp.?.forward(gen.xfm, &head_reference, head_ids, held_hidden, @intCast(st.chain.off0), .none, null, ids[0..kept]);
    if (head_out.logits.ctx != null) _ = try g.adopt(head_out.logits);
    if (head_out.hidden_next.ctx != null) _ = try g.adopt(head_out.hidden_next);
    if (head_out.rerank_x.ctx != null) _ = try g.adopt(head_out.rerank_x);

    const generated_before = gen.generated_ids.items.len;
    const completed_before = gen.completion_tokens;
    try gen.mtpRoundVerify(st);
    try ops.expectBitEqual(st.verify_logits, reference.logits);
    try ops.expectBitEqual(st.verify_hidden_all, reference.all);
    const result = (try gen.mtpRoundFinish(a, st)) orelse return error.ExpectedHistoryResult;
    defer a.free(result.tokens);
    try testing.expectEqual(accepted, result.accepted_tokens);
    try testing.expectEqualSlices(u32, ids[0..kept], result.tokens);
    try testing.expectEqualSlices(u32, ids[0..kept], gen.generated_ids.items[generated_before..]);
    try testing.expectEqual(completed_before + kept, gen.completion_tokens);
    try testing.expectEqual(next, gen.next_token_id);
    try testing.expectEqual(kept, st.retained);
    try expectHistoryKV(gen.ctx.cache, want_cache);
    try ops.expectBitEqual(gen.last_hidden, want_hidden);
    const stash = gen.mtp_hist_stash orelse return error.ExpectedHistoryStash;
    try testing.expectEqual(@as(usize, kept), stash.n);
    try testing.expectEqualSlices(u32, ids[0..kept], stash.host_ids[0..stash.n]);
    try ops.expectBitEqual(stash.hidden, held_hidden);
    try gen.mtpApplyStash();
    try testing.expect(gen.mtp_hist_stash == null);
    try expectHistoryKV(&gen.mtp_cache.?.plugin.kv, &head_reference.plugin.kv);
    try testing.expectEqual(head_reference.plugin.next_head, gen.mtp_cache.?.plugin.next_head);
    return accepted;
}

test "plugin MTP history: cold and carried owned proposals verify zero partial full and recover native head state" {
    const a = testing.allocator;
    var corpus = history_env.override(.MLX_SERVE_LOOKUP_CORPUS, null);
    defer corpus.restore();
    var readonly = history_env.override(.MLX_SERVE_LOOKUP_READ_ONLY, "1");
    defer readonly.restore();
    var unset = history_env.override(.MLX_SERVE_LOOKUP_STORE, null);
    defer unset.restore();
    const bounds = depth_bounds.active;
    depth_bounds.active = .{ .min = 3, .max = 3 };
    defer depth_bounds.active = bounds;
    const predraft = history_trace.mtp_predraft_override;
    history_trace.mtp_predraft_override = false;
    defer history_trace.mtp_predraft_override = predraft;
    var rig: HistoryRig = undefined;
    try rig.open();
    defer rig.close();
    for ([_]u32{ 0, 1, 3 }) |want_accepted| {
        var gen = try rig.generator(&history_query);
        defer gen.deinit(a);
        try testing.expect(gen.mtp.? == .plugin);
        try testing.expect(gen.external_history == null);
        gen.mtp_lookup_streak = true;
        gen.mtp_round_ema = 0;
        var snapshot = try gen.ctx.cache.snapshot();
        defer snapshot.deinit();
        var proposal = [_]u32{ gen.next_token_id, 0, 0, 0 };
        // Derive each proposal from real target logits at the same four-row
        // geometry. Corrupt only the requested reject position, never logits.
        for (0..3) |i| {
            var probe = try HistoryReplay.run(gen.xfm, &snapshot, gen.ctx.moe_seq_offset.*, &proposal, true);
            defer probe.deinit();
            var am = try history_verify.verifyArgmax(probe.logits, gen.sampling.suppress_mask, gen.xfm.s);
            defer am.deinit();
            try mlx.check(mlx.mlx_array_eval(am.lazy()));
            const target: u32 = @intCast((try am.ids(4))[i]);
            proposal[1 + i] = if (i == want_accepted) (target + 1) % 256 else target;
        }
        var document: [17]u32 = undefined;
        @memcpy(document[0..16], &history_query);
        document[16] = gen.next_token_id;
        var name: [32]u8 = undefined;
        const path = try rig.store(try std.fmt.bufPrint(&name, "accept-{d}", .{want_accepted}), &document, proposal[1..]);
        defer a.free(path);
        {
            var store = history_env.override(.MLX_SERVE_LOOKUP_STORE, path.ptr);
            defer store.restore();
            if (want_accepted != 0) {
                history_trace.mtp_predraft_override = true;
                try gen.mtpMaybePreDraft(a);
                history_trace.mtp_predraft_override = false;
                const carried = gen.mtp_pre_draft orelse return error.ExpectedCarriedHistory;
                try testing.expect(carried.lookup and carried.external_history);
                try testing.expectEqualSlices(u32, proposal[1..], carried.drafts);
                // Another request-local query invalidates/reuses the borrowed
                // candidate before the carried flight reaches its verifier.
                const replacement = (try gen.external_history.?.match(history_other_query[0..15], 95, 3)) orelse return error.ExpectedReplacementCandidate;
                try testing.expectEqualSlices(u32, history_native[0..3], replacement.draft);
                try testing.expectEqualSlices(u32, proposal[1..], gen.mtp_pre_draft.?.drafts);
            }
            var st = try historyBegin(&gen);
            defer st.deinit(a);
            try testing.expect(st.chain.lookup and st.chain.external_history);
            try testing.expectEqual(@as(u32, 3), st.chain.m);
            try testing.expectEqualSlices(u32, proposal[1..], st.chain.drafts);
            _ = try historyFinish(&gen, &st, want_accepted);
            try testing.expectEqual(@as(u32, 1), gen.external_history.?.rounds);
            try testing.expectEqual(want_accepted, gen.external_history.?.accepted);
            try testing.expectEqual(@as(u64, 0), gen.mtp_attempted);
        }
        // Disable only the external source: stock lookup and the actual native
        // head remain enabled. The short unique prompt has no admitted incumbent.
        var recovery = try historyBegin(&gen);
        defer recovery.deinit(a);
        try testing.expect(!recovery.chain.lookup and !recovery.chain.external_history);
        _ = try historyFinish(&gen, &recovery, null);
        try testing.expectEqual(@as(u64, 1), gen.mtp_attempted);
        try testing.expectEqual(@as(u32, 1), gen.external_history.?.rounds);
        try testing.expectEqual(want_accepted, gen.external_history.?.accepted);
    }
}

test "plugin MTP history: disabled source allocates no request through native verification and predraft" {
    const a = testing.allocator;
    var store = history_env.override(.MLX_SERVE_LOOKUP_STORE, null);
    defer store.restore();
    var corpus = history_env.override(.MLX_SERVE_LOOKUP_CORPUS, null);
    defer corpus.restore();
    const bounds = depth_bounds.active;
    depth_bounds.active = .{ .min = 3, .max = 3 };
    defer depth_bounds.active = bounds;
    const predraft = history_trace.mtp_predraft_override;
    history_trace.mtp_predraft_override = false;
    defer history_trace.mtp_predraft_override = predraft;
    var rig: HistoryRig = undefined;
    try rig.open();
    defer rig.close();
    var gen = try rig.generator(&history_query);
    defer gen.deinit(a);
    var st = try historyBegin(&gen);
    defer st.deinit(a);
    try testing.expect(!st.chain.lookup);
    _ = try historyFinish(&gen, &st, null);
    history_trace.mtp_predraft_override = true;
    try gen.mtpMaybePreDraft(a);
    try testing.expect(gen.mtp_pre_draft != null);
    try testing.expect(!gen.mtp_pre_draft.?.lookup);
    try testing.expect(gen.external_history == null);
    try testing.expect(rig.b.xfm.token_history == null);
}
