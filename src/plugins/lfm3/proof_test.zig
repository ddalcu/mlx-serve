//! The lfm3 table against mlx-vlm's LFM3 (`tests/dump_lfm3_fixtures.py`, f32 on the CPU): a tiny random pack with conv,
//! full and sliding layers, prefilled whole and in chunks past the window, then decoded step by step.

const std = @import("std");
const plugin = @import("plugin.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");
const fixture = @import("../../sdk/testing/fixture.zig");
const parity = @import("../../sdk/testing/parity.zig");

const testing = std.testing;
const VOCAB = 128;
const prompt = [_]u32{ 1, 17, 42, 125, 3, 99, 7, 80, 64, 5, 122, 31 };
const decode = [_]u32{ 77, 40, 9, 126 };

fn refRows(comptime name: []const u8) []const f32 {
    const bytes = @embedFile("fixtures/tiny/" ++ name ++ ".f32");
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// One forward of `tokens` from `pos` through the slot's cache; its logits rows as f32.
fn step(a: std.mem.Allocator, pack: *conformance.Loaded, cache: *test_cache.TestCache, tokens: []const u32, pos: u32, logits: abi.Logits) ![]f32 {
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const positions = [_]u32{pos};
    var out: abi.ForwardOut = .{};
    const mode: abi.Mode = if (tokens.len == 1) .decode else .prefill;
    const v = pack.table.forward(pack.model, &.{ .mode = mode, .logits = logits, .rows = @intCast(tokens.len), .tokens = .of(tokens), .slots = .of(&slot), .positions = .of(&positions), .caches = .of(&h), .cache_ops = &test_cache.ops }, &out);
    try testing.expect(v.isOk());
    if (out.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = out.hidden.ctx });
    return conformance.consumeF32(a, out.logits);
}

/// Bar: bf16 activations against an f32 reference (mlx-vlm's own bf16 run sits below it): every row's cosine above
/// 0.999 and no row whose margin decides it picks another token.
fn expectRows(ours: []const f32, ref: []const f32) !void {
    try testing.expectEqual(ref.len, ours.len);
    const rows = ref.len / VOCAB;
    const cmp = parity.qwen4CompareRows(ours, ref, rows, VOCAB);
    const misses = parity.decidedMisses(ours, ref, rows, VOCAB);
    if (cmp.min_cos <= 0.999 or misses > 0) std.debug.print("[lfm3 proof] {d} rows: min cos {d:.6}, decided misses {d}\n", .{ rows, cmp.min_cos, misses });
    try testing.expect(cmp.min_cos > 0.999 and misses == 0);
}

/// The staged pack's path and directory live in `arena` (the conformance stage never frees them).
fn open(arena: std.mem.Allocator) !conformance.Loaded {
    const fx: fixture.Fixture = .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile("fixtures/tiny/config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile("fixtures/tiny/model.safetensors") },
    } };
    return conformance.Loaded.open(testing.io, arena, plugin.plugin.name, &plugin.table, fx);
}

test "lfm3 proof: one prefill and four decode steps match mlx-vlm's LFM3" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const pre = try step(a, &pack, &cache, &prompt, 0, .all_rows);
    defer a.free(pre);
    try expectRows(pre, refRows("prefill_logits"));
    for (decode, 0..) |t, i| {
        const row = try step(a, &pack, &cache, &.{t}, @intCast(prompt.len + i), .last_row);
        defer a.free(row);
        try expectRows(row, refRows("decode_logits")[i * VOCAB ..][0..VOCAB]);
    }
}

test "lfm3 proof: a prefill in chunks across the sliding window and the conv window matches one prefill" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    const ref = refRows("prefill_logits");
    var at: usize = 0;
    for ([_]usize{ 3, 6, 3 }) |n| {
        const rows = try step(a, &pack, &cache, prompt[at .. at + n], @intCast(at), .all_rows);
        defer a.free(rows);
        try expectRows(rows, ref[at * VOCAB .. (at + n) * VOCAB]);
        at += n;
    }
}

// ── LFM3-VL ──

const media = @import("media.zig");
const vision_fx = "fixtures/tiny_vl/";

fn openVl(arena: std.mem.Allocator) !conformance.Loaded {
    const fx: fixture.Fixture = .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile(vision_fx ++ "config.json") },
        .{ .name = "processor_config.json", .bytes = @embedFile(vision_fx ++ "processor_config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile(vision_fx ++ "model.safetensors") },
    } };
    return conformance.Loaded.open(testing.io, arena, plugin.plugin.name, &plugin.table, fx);
}

fn f32s(comptime name: []const u8) []const f32 {
    const bytes = @embedFile(vision_fx ++ name);
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// `prepare`'s pixel values against the reference processor's; the item, its pixels freed.
fn expectPrepared(pack: *conformance.Loaded, in: abi.MediaInput, comptime ref: []const u8, grid: [3]u32, rows: u32) !abi.MediaItem {
    var item: abi.MediaItem = .{};
    try testing.expect(plugin.table.media.?.prepare(&pack.parsed, &in, 1 << 30, &item).isOk());
    try testing.expectEqual(grid, item.grid);
    try testing.expectEqual(rows, item.rows);
    const ours = try conformance.consumeF32(testing.allocator, .{ .ctx = item.pixels.ctx });
    defer testing.allocator.free(ours);
    item.pixels = .{};
    const want = f32s(ref);
    try testing.expectEqual(want.len, ours.len);
    // Pillow's bicubic rounds each pass to uint8, as ours does: equal up to f32 normalization.
    var worst: f32 = 0;
    for (ours, want) |x, y| worst = @max(worst, @abs(x - y));
    try testing.expect(worst < 1e-5);
    return item;
}

/// Bar: the tower runs bf16 against the f32 reference; every row's cosine above 0.999.
fn expectRowsClose(ours: []const f32, ref: []const f32, width: usize) !void {
    try testing.expectEqual(ref.len, ours.len);
    const cmp = parity.qwen4CompareRows(ours, ref, ref.len / width, width);
    if (cmp.min_cos <= 0.999) std.debug.print("[lfm3 proof] tower rows: min cos {d:.6}\n", .{cmp.min_cos});
    try testing.expect(cmp.min_cos > 0.999);
}

test "lfm3 proof: LFM3-VL prepares an image and a clip as mlx-vlm's processors do" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try openVl(arena.allocator());
    defer pack.close();
    const image = @embedFile(vision_fx ++ "image.rgb");
    _ = try expectPrepared(&pack, .{ .kind = .image, .rgb = .of(image), .width = 46, .height = 30, .frames = 1 }, "image_pixels.f32", .{ 1, 8, 12 }, 24);
    const clip = @embedFile(vision_fx ++ "clip.rgb");
    const item = try expectPrepared(&pack, .{ .kind = .video, .rgb = .of(clip), .width = 26, .height = 18, .frames = 3 }, "clip_pixels.f32", .{ 2, 4, 6 }, 12);
    try testing.expectEqual(@as(u32, 3), item.frames);
    try testing.expectEqual(@as(f32, 2.0), item.fps);
}

test "lfm3 proof: LFM3-VL's tower rows and the trunk over them match mlx-vlm's" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try openVl(arena.allocator());
    defer pack.close();
    const hooks = plugin.table.media.?;
    var rows_ref: ?abi.Array = null;
    for ([_]struct { []const u8, abi.MediaKind, u32, u32, u32 }{
        .{ @embedFile(vision_fx ++ "image.rgb"), .image, 46, 30, 1 },
        .{ @embedFile(vision_fx ++ "clip.rgb"), .video, 26, 18, 3 },
    }, [_][]const f32{ f32s("image_rows.f32"), f32s("clip_rows.f32") }) |c, ref| {
        var item: abi.MediaItem = .{};
        try testing.expect(hooks.prepare(&pack.parsed, &.{ .kind = c[1], .rgb = .of(c[0]), .width = c[2], .height = c[3], .frames = c[4] }, 1 << 30, &item).isOk());
        defer _ = mlx.mlx_array_free(.{ .ctx = item.pixels.ctx });
        var out: abi.Array = .{};
        try testing.expect(hooks.tower(pack.model, &item, &out).isOk());
        var dup = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&dup, .{ .ctx = out.ctx });
        const ours = try conformance.consumeF32(a, .{ .ctx = dup.ctx });
        defer a.free(ours);
        if (c[1] == .image) rows_ref = out else _ = mlx.mlx_array_free(.{ .ctx = out.ctx });
        try expectRowsClose(ours, ref, 32);
    }

    // The image's rows spliced at its placeholders, through the trunk (`[1, n, hidden]`, as the engine hands them).
    defer _ = mlx.mlx_array_free(.{ .ctx = rows_ref.?.ctx });
    var rows3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows3);
    try mlx.check(mlx.mlx_expand_dims(&rows3, .{ .ctx = rows_ref.?.ctx }, 0, pack.stream));
    const rows: abi.Array = .{ .ctx = rows3.ctx };
    const vl_prompt = [_]u32{ 1, 17, 122 } ++ @as([24]u32, @splat(120)) ++ [_]u32{ 123, 42, 99, 5 };
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const pos = [_]u32{0};
    var fo: abi.ForwardOut = .{};
    try testing.expect(pack.table.forward(pack.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = vl_prompt.len, .tokens = .of(&vl_prompt), .slots = .of(&slot), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &test_cache.ops, .media = rows }, &fo).isOk());
    if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
    const logits = try conformance.consumeF32(a, fo.logits);
    defer a.free(logits);
    try expectRows(logits, f32s("prefill_logits.f32"));
}

test "lfm3 proof: a clip's run stamps each frame group and holds its rows" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try openVl(arena.allocator());
    defer pack.close();
    const item: abi.MediaItem = .{ .kind = .video, .rows = 12, .grid = .{ 2, 4, 6 }, .fps = 2.0, .frames = 3 };
    // A stand-in tokenizer: one id per byte of the text, offset past the vocabulary's specials.
    const Bytes = struct {
        fn encode(_: ?*anyopaque, text: abi.Str, out: [*]u32, cap: u32) callconv(.c) u32 {
            if (text.len > cap) return 0;
            for (text.slice(), 0..) |ch, i| out[i] = 1000 + @as(u32, ch);
            return @intCast(text.len);
        }
    };
    const tok: abi.TextTokens = .{ .ctx = null, .encode = &Bytes.encode };
    var buf: [128]u32 = undefined;
    const n = plugin.table.media.?.placeholder_rows.?(&pack.parsed, &item, &tok, &buf, buf.len);
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(testing.allocator);
    // Groups (0,1) and (2,2) at 2 fps: 0.25 s and 1.0 s, as the reference's `<{t:.1f} seconds>`.
    for ([_][]const u8{ "<0.2 seconds>", "<1.0 seconds>" }) |label| {
        for (label) |ch| try want.append(testing.allocator, 1000 + @as(u32, ch));
        try want.append(testing.allocator, 122);
        try want.appendNTimes(testing.allocator, 121, 6);
        try want.append(testing.allocator, 123);
    }
    try testing.expectEqualSlices(u32, want.items, buf[0..n]);
}

// ── LFM3-Audio ──

const audio_fx = "fixtures/tiny_audio/";

fn openAudio(arena: std.mem.Allocator) !conformance.Loaded {
    const fx: fixture.Fixture = .{ .files = &.{
        .{ .name = "config.json", .bytes = @embedFile(audio_fx ++ "config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile(audio_fx ++ "model.safetensors") },
    } };
    return conformance.Loaded.open(testing.io, arena, plugin.plugin.name, &plugin.table, fx);
}

fn audioF32s(comptime name: []const u8) []const f32 {
    const bytes = @embedFile(audio_fx ++ name);
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

test "lfm3 proof: LFM3-Audio's log-mel, encoder rows and the trunk over them match mlx-vlm's" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try openAudio(arena.allocator());
    defer pack.close();
    const hooks = plugin.table.media.?;
    const clip = audioF32s("clip.f32");
    var item: abi.MediaItem = .{};
    try testing.expect(hooks.prepare(&pack.parsed, &.{ .kind = .audio, .samples = .of(clip), .sample_rate = 16000 }, 1 << 30, &item).isOk());
    defer _ = mlx.mlx_array_free(.{ .ctx = item.pixels.ctx });
    try testing.expectEqual(@as(u32, 49), item.grid[0]);
    try testing.expectEqual(@as(u32, 7), item.rows);

    // The features alone: f32 against the float64 reference, within f32 rounding of the log.
    const m: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
    {
        var g = @import("../../mlx/ops.zig").Graph.init(a, pack.stream);
        defer g.deinit();
        const feats = try m.speech.?.features(&g, .{ .ctx = item.pixels.ctx }, 49);
        const ours = try conformance.consumeF32(a, .{ .ctx = g.escape(feats).ctx });
        defer a.free(ours);
        const want = audioF32s("features.f32");
        try testing.expectEqual(want.len, ours.len);
        var worst: f32 = 0;
        for (ours, want) |x, y| worst = @max(worst, @abs(x - y));
        if (worst > 1e-3) std.debug.print("[lfm3 proof] log-mel max |ours - ref| = {e}\n", .{worst});
        try testing.expect(worst <= 1e-3);
    }

    var out: abi.Array = .{};
    try testing.expect(hooks.tower(pack.model, &item, &out).isOk());
    var dup = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&dup, .{ .ctx = out.ctx });
    const ours = try conformance.consumeF32(a, .{ .ctx = dup.ctx });
    defer a.free(ours);
    try expectRowsClose(ours, audioF32s("rows.f32"), 32);

    defer _ = mlx.mlx_array_free(.{ .ctx = out.ctx });
    var rows3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows3);
    try mlx.check(mlx.mlx_expand_dims(&rows3, .{ .ctx = out.ctx }, 0, pack.stream));
    const audio_prompt = [_]u32{ 1, 17, 42 } ++ @as([7]u32, @splat(110)) ++ [_]u32{ 99, 5 };
    var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
    defer cache.deinit();
    var h = [_]abi.Handle{cache.handle()};
    const slot = [_]u32{0};
    const pos = [_]u32{0};
    var fo: abi.ForwardOut = .{};
    try testing.expect(pack.table.forward(pack.model, &.{ .mode = .prefill, .logits = .all_rows, .rows = audio_prompt.len, .tokens = .of(&audio_prompt), .slots = .of(&slot), .positions = .of(&pos), .caches = .of(&h), .cache_ops = &test_cache.ops, .media = .{ .ctx = rows3.ctx } }, &fo).isOk());
    if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
    const logits = try conformance.consumeF32(a, fo.logits);
    defer a.free(logits);
    try expectRows(logits, audioF32s("prefill_logits.f32"));
}

test "lfm3 proof: the depth decoder's frames (plain and penalized) and a frame's input row match mlx-vlm's" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try openAudio(arena.allocator());
    defer pack.close();
    const m: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
    const d = &m.depth.?;
    const raw_hidden = mlx.mlx_array_new_data(audioF32s("depth_hidden.f32").ptr, &[_]c_int{ 1, 32 }, 2, .float32);
    defer _ = mlx.mlx_array_free(raw_hidden);
    const hidden: @import("../../mlx/ops.zig").Array = .{ .ctx = raw_hidden.ctx };
    var codes: [3]u32 = undefined;
    // Greedy over the reference's fixed reveal order; the penalty reads each codebook's own history.
    try d.frame(hidden, .{ .temperature = 0, .order = &.{ 2, 0, 1 }, .key = 0 }, &.{}, &codes);
    try testing.expectEqualSlices(u32, &.{ 3, 6, 8 }, &codes);
    try d.frame(hidden, .{ .temperature = 0, .repetition_penalty = 8.0, .order = &.{ 2, 0, 1 }, .key = 0 }, &.{ 3, 9, 16, 5, 9, 2 }, &codes);
    try testing.expectEqualSlices(u32, &.{ 11, 6, 8 }, &codes);
    const row = try d.frameRows(&.{ 3, 6, 8 });
    const ours = try conformance.consumeF32(a, .{ .ctx = row.ctx });
    defer a.free(ours);
    try expectRowsClose(ours, audioF32s("frame_row.f32"), 32);
}
