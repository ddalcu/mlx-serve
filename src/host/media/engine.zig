//! Decoded image descriptors for an engine that owns its image encoder and placeholder expansion.

const std = @import("std");
const abi = @import("../../abi/engine.zig");
const chat_mod = @import("../chat/chat.zig");
const MediaItem = @import("placeholders.zig").MediaItem;
const MediaSpan = @import("../../sdk/kv/media_span.zig").MediaSpan;
const MediaPieces = @import("pieces.zig").MediaPieces;
const MultimodalPrompt = @import("encode.zig").MultimodalPrompt;
const multimodalFailure = @import("encode.zig").multimodalFailure;
const testing = std.testing;

/// An external engine widens its own image atoms. Keep the host prompt intact and key each atom by its RGB.
pub fn prepare(allocator: std.mem.Allocator, items: []const MediaItem, prompt_ids: []const u32, atom: []const u32) !MultimodalPrompt {
    if (atom.len == 0) return error.MediaPlaceholderMismatch;
    for (items) |item| switch (item) {
        .image => |images| {
            if (images.len != 1) return error.MediaPlaceholderMismatch;
            const img = images[0];
            const pixels = std.math.mul(usize, img.width, img.height) catch return error.InvalidImagePixels;
            const bytes = std.math.mul(usize, pixels, 3) catch return error.InvalidImagePixels;
            if (bytes == 0 or bytes != img.pixels.len) return error.InvalidImagePixels;
        },
        .video => return error.VideoUnsupported,
        .audio => return error.AudioUnsupported,
    };
    var pieces = try MediaPieces.init(allocator, items);
    defer pieces.deinit(allocator);
    var out: MultimodalPrompt = .{};
    errdefer out.deinit(allocator);
    out.media = try allocator.alloc(MediaSpan, items.len);
    out.engine_media = try allocator.alloc(abi.Media, items.len);
    var at: usize = 0;
    var k: usize = 0;
    while (std.mem.indexOfPos(u32, prompt_ids, at, atom)) |start| {
        if (k == items.len) return error.MediaPlaceholderMismatch;
        const img = items[k].image[0];
        const key = pieces.item_keys[k];
        out.media[k] = .{ .start = @intCast(start), .key = key };
        out.engine_media[k] = .{
            .kind = .image,
            .at = @intCast(start),
            .width = img.width,
            .height = img.height,
            .data = abi.span(u8, img.pixels),
            .hash = key,
        };
        k += 1;
        at = start + atom.len;
    }
    if (k != items.len) return error.MediaPlaceholderMismatch;
    return out;
}

test "engine image transport preserves atoms and prepares ordered pixel hashes without MLX" {
    const a = testing.allocator;
    const images = [_]chat_mod.ImageData{
        .{ .pixels = &.{ 1, 2, 3, 4, 5, 6 }, .width = 2, .height = 1 },
        .{ .pixels = &.{ 7, 8, 9 }, .width = 1, .height = 1 },
    };
    const items = [_]MediaItem{ .{ .image = images[0..1] }, .{ .image = images[1..2] } };
    const ids = [_]u32{ 1, 40, 41, 42, 2, 40, 41, 42, 3 };
    const atom = [_]u32{ 40, 41, 42 };
    var mm = try prepare(a, &items, &ids, &atom);
    defer mm.deinit(a);
    try testing.expect(mm.embeddings == null);
    try testing.expect(mm.mrope.pos == null);
    try testing.expectEqual(@as(usize, 0), mm.rows);
    try testing.expectEqual(@as(usize, 2), mm.engine_media.len);
    for (mm.engine_media, mm.media, images, 0..) |m, span, image, i| {
        try testing.expectEqual(.image, m.kind);
        try testing.expectEqual(@as(u32, @intCast(1 + 4 * i)), m.at);
        try testing.expectEqual(span.start, m.at);
        try testing.expectEqual(image.width, m.width);
        try testing.expectEqual(image.height, m.height);
        try testing.expectEqualSlices(u8, image.pixels, m.data.slice());
        try testing.expectEqual(span.key, m.hash);
        try testing.expect(m.hash != 0);
    }
    try testing.expect(mm.engine_media[0].hash != mm.engine_media[1].hash);
    var again = try prepare(a, &items, &ids, &atom);
    defer again.deinit(a);
    try testing.expectEqual(mm.engine_media[0].hash, again.engine_media[0].hash);
    const reshaped = [_]chat_mod.ImageData{.{ .pixels = images[0].pixels, .width = 1, .height = 2 }};
    var shape = try prepare(a, &.{.{ .image = &reshaped }}, ids[0..4], &atom);
    defer shape.deinit(a);
    try testing.expect(shape.engine_media[0].hash != mm.engine_media[0].hash);
}

test "engine image transport rejects missing extra truncated and malformed image atoms" {
    const a = testing.allocator;
    const image = [_]chat_mod.ImageData{.{ .pixels = &.{ 1, 2, 3 }, .width = 1, .height = 1 }};
    const items = [_]MediaItem{.{ .image = &image }};
    const atom = [_]u32{ 40, 41, 42 };
    for ([_][]const u32{ &.{1}, &.{ 40, 41 }, &.{ 40, 41, 42, 40, 41, 42 } }) |ids| {
        try testing.expectError(error.MediaPlaceholderMismatch, prepare(a, &items, ids, &atom));
    }
    try testing.expectError(error.MediaPlaceholderMismatch, prepare(a, &items, &atom, &.{}));
    const bad = [_]chat_mod.ImageData{.{ .pixels = &.{ 1, 2, 3 }, .width = 2, .height = 1 }};
    try testing.expectError(error.InvalidImagePixels, prepare(a, &.{.{ .image = &bad }}, &atom, &atom));
    const audio = chat_mod.AudioData{ .samples = &.{} };
    try testing.expectError(error.AudioUnsupported, prepare(a, &.{.{ .audio = &audio }}, &atom, &atom));
    try testing.expectEqual(@as(u16, 400), multimodalFailure(error.InvalidImagePixels).code);
}

test "engine image transport public preparation uses the engine's atom and bypasses MLX encoding" {
    const Fixture = struct {
        const atom = "<start><image><end>";
        const tokens = [_]u32{ 40, 41, 42 };
        fn tokenize(_: *anyopaque, text: abi.Str, specials: bool, out: [*]u32, cap: usize) callconv(.c) usize {
            if (specials or !std.mem.eql(u8, atom, text.slice())) return 0;
            if (cap >= tokens.len) @memcpy(out[0..tokens.len], &tokens);
            return tokens.len;
        }
    };
    const a = testing.allocator;
    var table = @import("../../testing/fake_engine.zig").own_thread;
    table.tokenize = Fixture.tokenize;
    var engine: @import("../engines/served.zig").Served = undefined;
    engine.info = std.mem.zeroInit(abi.Info, .{});
    engine.info.caps.images = true;
    engine.info.image_placeholder = abi.Str.of(Fixture.atom);
    engine.table = &table;
    var handle: u8 = 0;
    engine.handle = &handle;
    var lm: @import("../registry/registry.zig").LoadedModel = undefined;
    lm.engine = &engine;
    lm.vision_encoder = null;
    const config = @import("../../sdk/config/model.zig").ModelConfig{ .qwen_vision = true };
    const image = [_]chat_mod.ImageData{.{ .pixels = &.{ 11, 22, 33 }, .width = 1, .height = 1 }};
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "look", .images = &image }};
    const ids = [_]u32{ 1, 40, 41, 42, 2 };
    var prompt = try a.dupe(u32, &ids);
    defer a.free(prompt);
    const original = prompt.ptr;
    var mm = try @import("encode.zig").prepareMultimodalPrompt(a, &lm, &config, &messages, &prompt);
    defer mm.deinit(a);
    try testing.expect(mm.embeddings == null);
    try testing.expectEqual(@as(usize, 1), mm.engine_media.len);
    try testing.expectEqual(@as(u32, 1), mm.engine_media[0].at);
    try testing.expectEqualSlices(u32, &ids, prompt);
    try testing.expect(original == prompt.ptr);
    engine.info.caps.images = false;
    try testing.expectError(error.ImagesUnsupported, @import("encode.zig").prepareMultimodalPrompt(a, &lm, &config, &messages, &prompt));
    try testing.expectEqual(@as(u16, 400), multimodalFailure(error.ImagesUnsupported).code);
}

test "engine image transport releases partially allocated descriptor preparations" {
    const Case = struct {
        fn run(a: std.mem.Allocator) !void {
            const image = [_]chat_mod.ImageData{.{ .pixels = &.{ 1, 2, 3 }, .width = 1, .height = 1 }};
            var mm = try prepare(a, &.{.{ .image = &image }}, &.{ 1, 40, 41, 42 }, &.{ 40, 41, 42 });
            defer mm.deinit(a);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, Case.run, .{});
}
