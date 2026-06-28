//! PNG encoding for the native image-generation endpoint, via stb_image_write
//! (vendored `lib/stb_image_write.h`). Encodes an RGB8 buffer to PNG bytes in
//! memory (no temp file) using the `_to_func` callback variant.

const std = @import("std");

const StbiWriteFunc = *const fn (context: ?*anyopaque, data: ?*anyopaque, size: c_int) callconv(.c) void;
extern "c" fn stbi_write_png_to_func(
    func: StbiWriteFunc,
    context: ?*anyopaque,
    w: c_int,
    h: c_int,
    comp: c_int,
    data: ?*const anyopaque,
    stride_in_bytes: c_int,
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

/// Encode `rgb` (interleaved RGB8, `w*h*3` bytes, row-major) as PNG bytes.
/// Caller owns the returned slice.
pub fn encodeRgb(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32) ![]u8 {
    std.debug.assert(rgb.len == @as(usize, w) * @as(usize, h) * 3);
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(allocator);
    var sink = Sink{ .list = &list, .allocator = allocator };
    const rc = stbi_write_png_to_func(writeCb, &sink, @intCast(w), @intCast(h), 3, rgb.ptr, @intCast(w * 3));
    if (rc == 0 or sink.err) return error.PngEncodeFailed;
    return list.toOwnedSlice(allocator);
}

test "encodeRgb produces a valid PNG" {
    const a = std.testing.allocator;
    // 2x2 RGB checkerboard.
    const rgb = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    const png = try encodeRgb(a, &rgb, 2, 2);
    defer a.free(png);
    try std.testing.expect(png.len > 8);
    // PNG magic: 89 50 4E 47 0D 0A 1A 0A
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, png[0..8]);
    // IHDR chunk type at offset 12.
    try std.testing.expectEqualSlices(u8, "IHDR", png[12..16]);
    // Ends with IEND.
    try std.testing.expectEqualSlices(u8, "IEND", png[png.len - 8 .. png.len - 4]);
}
