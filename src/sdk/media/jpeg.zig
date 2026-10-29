//! JPEG encoding for opt-in media-gen preview frames, via stb_image_write
//! (vendored `lib/stb_image_write.h`). Encodes an RGB8 buffer to JPEG bytes in
//! memory (no temp file) using the `_to_func` callback variant.

const std = @import("std");

const StbiWriteFunc = *const fn (context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void;
extern "c" fn stbi_write_jpg_to_func(
    func: StbiWriteFunc,
    context: ?*anyopaque,
    w: c_int,
    h: c_int,
    comp: c_int,
    data: ?*const anyopaque,
    quality: c_int,
) c_int;

const Sink = struct {
    list: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    err: bool = false,
};

fn writeCb(context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void {
    const sink: *Sink = @ptrCast(@alignCast(context.?));
    if (size <= 0 or data == null) return;
    const bytes: [*]const u8 = @ptrCast(data.?);
    sink.list.appendSlice(sink.allocator, bytes[0..@intCast(size)]) catch {
        sink.err = true;
    };
}

/// Encode `rgb` (interleaved RGB8, `w*h*3` bytes, row-major) as JPEG bytes.
/// `quality` is 1–100 (stb's scale). Caller owns the returned slice.
pub fn encodeRgb(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32, quality: u8) ![]u8 {
    std.debug.assert(rgb.len == @as(usize, w) * @as(usize, h) * 3);
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var sink = Sink{ .list = &list, .allocator = allocator };
    const q: c_int = @intCast(@max(@as(u8, 1), @min(quality, 100)));
    const rc = stbi_write_jpg_to_func(writeCb, &sink, @intCast(w), @intCast(h), 3, rgb.ptr, q);
    if (rc == 0 or sink.err) return error.JpegEncodeFailed;
    return list.toOwnedSlice(allocator);
}

test "encodeRgb produces a valid JPEG" {
    const a = std.testing.allocator;
    const rgb = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    const jpg = try encodeRgb(a, &rgb, 2, 2, 72);
    defer a.free(jpg);
    try std.testing.expect(jpg.len > 4);
    try std.testing.expectEqual(@as(u8, 0xFF), jpg[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), jpg[1]);
    try std.testing.expectEqual(@as(u8, 0xFF), jpg[2]);
}
