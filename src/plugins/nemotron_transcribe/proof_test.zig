//! The nemotron_transcribe table over a tiny random pack (no reference oracle yet): a clip through `prepare`, the
//! perception module and a spliced prefill, with windows small enough that the speaker FIFO pops and the speaker cache
//! compresses.

const std = @import("std");
const plugin = @import("plugin.zig");
const config = @import("config.zig");
const perception = @import("perception.zig");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const conformance = @import("../../sdk/testing/conformance.zig");
const test_cache = @import("../../sdk/testing/cache_ops.zig");

const testing = std.testing;

fn open(arena: std.mem.Allocator) !conformance.Loaded {
    return conformance.Loaded.open(testing.io, arena, plugin.plugin.name, &plugin.table, try plugin.conformanceFixture(arena));
}

/// A deterministic two-tone clip: a low tone, then a higher one, with a gap of silence between.
fn clip(a: std.mem.Allocator, seconds: f32) ![]f32 {
    const n: usize = @intFromFloat(seconds * 16000);
    const out = try a.alloc(f32, n);
    for (out, 0..) |*x, i| {
        const t = @as(f32, @floatFromInt(i)) / 16000;
        const frac = t / seconds;
        x.* = if (frac > 0.4 and frac < 0.5) 0 else 0.3 * @sin(2 * std.math.pi * (if (frac < 0.45) @as(f32, 220) else 880) * t);
    }
    return out;
}

fn towerRows(pack: *conformance.Loaded, pcm: []const f32) !struct { item: abi.MediaItem, rows: abi.Array } {
    const hooks = plugin.table.media.?;
    var item: abi.MediaItem = .{};
    try testing.expect(hooks.prepare(&pack.parsed, &.{ .kind = .audio, .samples = .of(pcm), .sample_rate = 16000 }, 1 << 30, &item).isOk());
    var out: abi.Array = .{};
    try testing.expect(hooks.tower(pack.model, &item, &out).isOk());
    return .{ .item = item, .rows = out };
}

test "nemotron_transcribe proof: a clip's rows span its hops, finite, windowed through a FIFO pop and a cache compression" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const pcm = try clip(a, 4.0);
    defer a.free(pcm);
    const got = try towerRows(&pack, pcm);
    defer _ = mlx.mlx_array_free(.{ .ctx = got.item.pixels.ctx });
    // 4 s = 400 frames = 50 rows over 6-row windows: eight windows, past the FIFO (3) and the cache (20).
    try testing.expectEqual(@as(u32, 50), got.item.rows);
    const rows = try conformance.consumeF32(a, got.rows);
    defer a.free(rows);
    try testing.expectEqual(@as(usize, 50 * 64), rows.len);
    var energy: f32 = 0;
    for (rows) |x| {
        try testing.expect(std.math.isFinite(x));
        energy += x * x;
    }
    try testing.expect(energy > 0);
}

test "nemotron_transcribe proof: the tower is deterministic and hears the clip (another clip, other rows)" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const pcm = try clip(a, 1.5);
    defer a.free(pcm);
    const first = try towerRows(&pack, pcm);
    defer _ = mlx.mlx_array_free(.{ .ctx = first.item.pixels.ctx });
    const second = try towerRows(&pack, pcm);
    defer _ = mlx.mlx_array_free(.{ .ctx = second.item.pixels.ctx });
    const x = try conformance.consumeF32(a, first.rows);
    defer a.free(x);
    const y = try conformance.consumeF32(a, second.rows);
    defer a.free(y);
    try testing.expectEqualSlices(f32, x, y);

    const other = try a.dupe(f32, pcm);
    defer a.free(other);
    for (other, 0..) |*s, i| s.* *= if (i % 7 == 0) -1 else 0.5;
    const third = try towerRows(&pack, other);
    defer _ = mlx.mlx_array_free(.{ .ctx = third.item.pixels.ctx });
    const z = try conformance.consumeF32(a, third.rows);
    defer a.free(z);
    var diff: f32 = 0;
    for (x, z) |p, q| diff += @abs(p - q);
    try testing.expect(diff > 1e-3);
}

test "nemotron_transcribe proof: rows splice into a prefill at the audio placeholder and shift the logits" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const pcm = try clip(a, 1.0);
    defer a.free(pcm);
    const got = try towerRows(&pack, pcm);
    defer _ = mlx.mlx_array_free(.{ .ctx = got.item.pixels.ctx });
    defer _ = mlx.mlx_array_free(.{ .ctx = got.rows.ctx });
    const rows = got.item.rows;
    try testing.expectEqual(@as(u32, 13), rows);

    // The run is rows of the tokenizer's `<|audio|>`, which sits one past the tiny table, as in the real pack.
    const audio = pack.parsed.placeholder_ids.audio;
    try testing.expectEqual(@as(u32, 64), audio);
    var ids: [32]u32 = undefined;
    ids[0] = 5;
    ids[1] = 9;
    @memset(ids[2 .. 2 + rows], audio);
    ids[2 + rows] = 7;
    const prompt = ids[0 .. 3 + rows];

    var media3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(media3);
    try mlx.check(mlx.mlx_expand_dims(&media3, .{ .ctx = got.rows.ctx }, 0, pack.stream));

    // The engine hands its ids as a device array with `tokens` empty; a direct caller hands host tokens.
    const shape = [_]c_int{ 1, @intCast(prompt.len) };
    const device_ids = mlx.mlx_array_new_data(prompt.ptr, &shape, 2, .uint32);
    defer _ = mlx.mlx_array_free(device_ids);

    const Arm = enum { host_media, none, device_media };
    var logits: [3][]f32 = undefined;
    for ([_]Arm{ .host_media, .none, .device_media }, 0..) |arm, k| {
        var cache = try test_cache.TestCache.init(a, pack.parsed.shell.num_hidden_layers);
        defer cache.deinit();
        var h = [_]abi.Handle{cache.handle()};
        const slot = [_]u32{0};
        const pos = [_]u32{0};
        var fo: abi.ForwardOut = .{};
        const f: abi.Forward = .{
            .mode = .prefill,
            .logits = .last_row,
            .rows = @intCast(prompt.len),
            .tokens = if (arm == .device_media) .{} else .of(prompt),
            .ids = if (arm == .device_media) .{ .ctx = device_ids.ctx } else .{},
            .slots = .of(&slot),
            .positions = .of(&pos),
            .caches = .of(&h),
            .cache_ops = &test_cache.ops,
            .media = if (arm == .none) .{} else .{ .ctx = media3.ctx },
        };
        try testing.expect(pack.table.forward(pack.model, &f, &fo).isOk());
        if (fo.hidden.ctx != null) _ = mlx.mlx_array_free(.{ .ctx = fo.hidden.ctx });
        logits[k] = try conformance.consumeF32(a, fo.logits);
    }
    defer for (logits) |l| a.free(l);
    var diff: f32 = 0;
    for (logits[0], logits[1]) |p, q| diff += @abs(p - q);
    try testing.expect(diff > 1e-4);
    // The same rows spliced by either form of the ids give the same logits.
    try testing.expectEqualSlices(f32, logits[0], logits[2]);
}

test "nemotron_transcribe proof: a clip under one hop, or another rate, is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const hooks = plugin.table.media.?;
    var item: abi.MediaItem = .{};
    const short: [100]f32 = @splat(0.1);
    try testing.expect(!hooks.prepare(&pack.parsed, &.{ .kind = .audio, .samples = .of(&short), .sample_rate = 16000 }, 1 << 30, &item).isOk());
    const long: [3200]f32 = @splat(0.1);
    try testing.expect(!hooks.prepare(&pack.parsed, &.{ .kind = .audio, .samples = .of(&long), .sample_rate = 24000 }, 1 << 30, &item).isOk());
    try testing.expect(!hooks.prepare(&pack.parsed, &.{ .kind = .image }, 1 << 30, &item).isOk());
}

test "nemotron_transcribe proof: the speaker expert alone gives a probability per speaker for every 10 ms frame" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const pcm = try clip(a, 4.0);
    defer a.free(pcm);
    const hooks = plugin.table.media.?;
    var speakers: u32 = 0;
    var out: abi.Span(f32) = .{};
    try testing.expect(hooks.diarize.?(pack.model, .of(pcm), &speakers, &out).isOk());
    try testing.expectEqual(@as(u32, 8), speakers);
    // 4 s = 400 mel frames, past the 6-row windows, the FIFO and the cache.
    try testing.expectEqual(@as(usize, 400 * 8), out.slice().len);
    for (out.slice()) |x| try testing.expect(x >= 0 and x <= 1);
    // A second call replaces the first answer (plugin-owned), and the same clip hears the same speakers.
    const first = try a.dupe(f32, out.slice());
    defer a.free(first);
    try testing.expect(hooks.diarize.?(pack.model, .of(pcm), &speakers, &out).isOk());
    try testing.expectEqualSlices(f32, first, out.slice());
}

test "nemotron_transcribe proof: the CTC head times each word inside the clip, in order, and names who speaks" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var pack = try open(arena.allocator());
    defer pack.close();
    const loaded: *plugin.Loaded = @ptrCast(@alignCast(pack.model));
    const pcm = try clip(a, 4.0);
    defer a.free(pcm);
    // Words of 2, 1, 3 and 1 tokens in two turns (the tiny head reads 12 classes).
    var w0 = [_]u32{ 3, 5 };
    var w1 = [_]u32{5};
    var w2 = [_]u32{ 1, 7, 2 };
    var w3 = [_]u32{9};
    const words = [_]perception.Tower.TokenWord{ .{ .tag = 0, .ids = &w0 }, .{ .tag = 0, .ids = &w1 }, .{ .tag = 1, .ids = &w2 }, .{ .tag = 1, .ids = &w3 } };

    // Without speaker activity: one pass over every frame, no speaker named.
    const plain = try loaded.tower.alignTokens(pcm, &words, &.{}, 0);
    defer std.heap.c_allocator.free(plain);
    try testing.expectEqual(@as(usize, 4), plain.len);
    // Each speaker's words come in order (the turns are searched apart, over the same frames).
    for (plain, 0..) |x, i| {
        try testing.expect(x.end >= x.start and x.end <= 4.0 + 1.6);
        try testing.expectEqual(@as(i32, -1), x.speaker);
        if (i % 2 == 1) try testing.expect(x.start >= plain[i - 1].start);
    }

    // With the clip's activity, each turn is named for a column.
    const hooks = plugin.table.media.?;
    var speakers: u32 = 0;
    var act: abi.Span(f32) = .{};
    try testing.expect(hooks.diarize.?(pack.model, .of(pcm), &speakers, &act).isOk());
    const timed = try loaded.tower.alignTokens(pcm, &words, act.slice(), speakers);
    defer std.heap.c_allocator.free(timed);
    try testing.expectEqual(@as(usize, 4), timed.len);
    try testing.expectEqual(timed[0].speaker, timed[1].speaker);
    try testing.expectEqual(timed[2].speaker, timed[3].speaker);

    try testing.expectError(error.TokenOutOfRange, loaded.tower.alignTokens(pcm, &.{.{ .tag = -1, .ids = @constCast(&[_]u32{99}) }}, &.{}, 0));
    // No alignment tokenizer in the tiny pack: words are refused by name, not guessed.
    try testing.expectError(error.NoAlignmentTokenizer, loaded.tower.wordTimes(pcm, &.{"hello"}, &.{}, 0));
}

test {
    _ = perception;
}
