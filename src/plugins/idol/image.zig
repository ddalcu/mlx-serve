//! Idol's `preprocess_image`: resize so the short side is 1024, take the alpha (the image's own, or the background
//! remover's), erode it by one pixel, crop a square around the subject's box with a margin, resize to 1024 and
//! composite over black. Pillow's arithmetic throughout: premultiplied RGBA resizes, `MinFilter(3)`, `paste` blend.

const std = @import("std");
const qvis = @import("../../sdk/vision/qwen_vision.zig");

pub const IMAGE_SIZE: u32 = 1024;
/// How much room the square crop leaves around the subject, per `mode`.
pub const MARGIN_SYNTHETIC: f64 = 1.2;
pub const MARGIN_REAL: f64 = 1.7;

pub const Rgba = struct {
    pix: []u8,
    w: u32,
    h: u32,

    pub fn deinit(self: Rgba, a: std.mem.Allocator) void {
        a.free(self.pix);
    }
};

/// PIL's `MULDIV255`: round(x * y / 255).
fn muldiv255(x: u32, y: u32) u32 {
    const t = x * y + 128;
    return ((t >> 8) + t) >> 8;
}

/// Python's `round` of a positive value: ties to even.
fn roundEven(v: f64) u32 {
    const f = @floor(v);
    const d = v - f;
    const up = d > 0.5 or (d == 0.5 and @mod(f, 2) == 1);
    return @intFromFloat(if (up) f + 1 else f);
}

/// `Image.resize((w, h), LANCZOS)` of straight-alpha RGBA: Pillow resamples premultiplied RGBa and converts back.
/// The same size is a copy.
pub fn resize(a: std.mem.Allocator, src: []const u8, w: u32, h: u32, nw: u32, nh: u32) ![]u8 {
    if (w == nw and h == nh) return a.dupe(u8, src);
    const pre = try a.alloc(u8, src.len);
    defer a.free(pre);
    for (0..src.len / 4) |i| {
        const al: u32 = src[i * 4 + 3];
        for (0..3) |c| pre[i * 4 + c] = @intCast(muldiv255(src[i * 4 + c], al));
        pre[i * 4 + 3] = src[i * 4 + 3];
    }
    const out = try qvis.resizeInterleavedPil(a, pre, h, w, nh, nw, 4, .lanczos);
    errdefer a.free(out);
    for (0..out.len / 4) |i| {
        const al: u32 = out[i * 4 + 3];
        if (al == 0 or al == 255) continue;
        for (0..3) |c| out[i * 4 + c] = @intCast(@min(255, @as(u32, out[i * 4 + c]) * 255 / al));
    }
    return out;
}

/// Whether any pixel is not fully opaque (`has_real_alpha`).
pub fn hasRealAlpha(rgba: []const u8) bool {
    var i: usize = 3;
    while (i < rgba.len) : (i += 4) if (rgba[i] < 255) return true;
    return false;
}

/// The input resized so its short side is `size` (1024 as released).
pub fn toShortSide(a: std.mem.Allocator, src: []const u8, w: u32, h: u32, size: u32) !Rgba {
    const scale = @as(f64, @floatFromInt(size)) / @as(f64, @floatFromInt(@min(w, h)));
    const nw = @max(1, roundEven(@as(f64, @floatFromInt(w)) * scale));
    const nh = @max(1, roundEven(@as(f64, @floatFromInt(h)) * scale));
    return .{ .pix = try resize(a, src, w, h, nw, nh), .w = nw, .h = nh };
}

/// `ImageFilter.MinFilter(3)` on one plane: Pillow leaves the one-pixel border as it was.
fn minFilter3(a: std.mem.Allocator, plane: []u8, w: usize, h: usize) !void {
    if (w < 3 or h < 3) return;
    const src = try a.dupe(u8, plane);
    defer a.free(src);
    for (1..h - 1) |y| for (1..w - 1) |x| {
        var m: u8 = 255;
        for (0..3) |dy| {
            for (0..3) |dx| m = @min(m, src[(y + dy - 1) * w + x + dx - 1]);
        }
        plane[y * w + x] = m;
    };
}

/// The square RGB8 the encoders see (`size` x `size`, subject over black). `rgba` is the resized image;
/// `alpha`, when given, replaces its alpha channel (the background remover's matte, same size). Errors with
/// `EmptySubject` when no pixel is left.
pub fn prepare(a: std.mem.Allocator, rgba: Rgba, alpha: ?[]const u8, margin: f64, size: u32) ![]u8 {
    const w: usize = rgba.w;
    const h: usize = rgba.h;
    const img = try a.dupe(u8, rgba.pix);
    defer a.free(img);
    const plane = try a.alloc(u8, w * h);
    defer a.free(plane);
    for (0..w * h) |i| plane[i] = if (alpha) |al| al[i] else img[i * 4 + 3];
    try minFilter3(a, plane, w, h);
    for (0..w * h) |i| img[i * 4 + 3] = plane[i];

    var l: usize = w;
    var t: usize = h;
    var r: usize = 0;
    var b: usize = 0;
    var any = false;
    for (0..h) |y| for (0..w) |x| if (plane[y * w + x] != 0) {
        any = true;
        l = @min(l, x);
        r = @max(r, x);
        t = @min(t, y);
        b = @max(b, y);
    };
    if (!any) return error.EmptySubject;
    const cx = @as(f64, @floatFromInt(l + r)) / 2;
    const cy = @as(f64, @floatFromInt(t + b)) / 2;
    const half = @as(f64, @floatFromInt(@max(r - l, b - t))) / 2 * margin;
    const x0: i64 = @intFromFloat(cx - half); // int() truncates toward zero
    const y0: i64 = @intFromFloat(cy - half);
    const x1: i64 = @intFromFloat(cx + half);
    const y1: i64 = @intFromFloat(cy + half);
    const cw: usize = @intCast(@max(1, x1 - x0));
    const ch: usize = @intCast(@max(1, y1 - y0));
    const crop = try a.alloc(u8, cw * ch * 4);
    defer a.free(crop);
    @memset(crop, 0);
    for (0..ch) |y| for (0..cw) |x| {
        const sx = x0 + @as(i64, @intCast(x));
        const sy = y0 + @as(i64, @intCast(y));
        if (sx < 0 or sy < 0 or sx >= w or sy >= h) continue;
        const si = (@as(usize, @intCast(sy)) * w + @as(usize, @intCast(sx))) * 4;
        @memcpy(crop[(y * cw + x) * 4 ..][0..4], img[si..][0..4]);
    };
    const big = try resize(a, crop, @intCast(cw), @intCast(ch), size, size);
    defer a.free(big);
    const out = try a.alloc(u8, @as(usize, size) * size * 3);
    for (0..@as(usize, size) * size) |i| {
        const al: u32 = big[i * 4 + 3];
        for (0..3) |c| out[i * 3 + c] = @intCast(muldiv255(big[i * 4 + c], al));
    }
    return out;
}

// ── tests ──

const testing = std.testing;

test "idol image: the short side lands on 1024 and ties round to even" {
    try testing.expectEqual(@as(u32, 2), roundEven(2.5));
    try testing.expectEqual(@as(u32, 4), roundEven(3.5));
    try testing.expectEqual(@as(u32, 3), roundEven(3.4));
    const px = try testing.allocator.alloc(u8, 8 * 4 * 4);
    defer testing.allocator.free(px);
    @memset(px, 255);
    const r = try toShortSide(testing.allocator, px, 8, 4, IMAGE_SIZE);
    defer r.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 2048), r.w);
    try testing.expectEqual(@as(u32, 1024), r.h);
}

test "idol image: an alpha blob is eroded, cropped with the margin and composited over black" {
    const a = testing.allocator;
    const w = 64;
    const px = try a.alloc(u8, w * w * 4);
    defer a.free(px);
    for (0..w * w) |i| {
        const x = i % w;
        const y = i / w;
        const inside = x >= 20 and x < 40 and y >= 24 and y < 36;
        px[i * 4 ..][0..4].* = .{ 200, 100, 50, if (inside) 255 else 0 };
    }
    try testing.expect(hasRealAlpha(px));
    const out = try prepare(a, .{ .pix = px, .w = w, .h = w }, null, 1.2, IMAGE_SIZE);
    defer a.free(out);
    try testing.expectEqual(@as(usize, 1024 * 1024 * 3), out.len);
    // The corner is outside the subject (black); the centre is the subject's colour.
    try testing.expectEqual(@as(u8, 0), out[0]);
    const mid = (512 * 1024 + 512) * 3;
    try testing.expectEqualSlices(u8, &.{ 200, 100, 50 }, out[mid..][0..3]);
}

test "idol image: an empty alpha is refused by name" {
    const a = testing.allocator;
    const px = try a.alloc(u8, 16 * 16 * 4);
    defer a.free(px);
    @memset(px, 0);
    try testing.expectError(error.EmptySubject, prepare(a, .{ .pix = px, .w = 16, .h = 16 }, null, 1.2, IMAGE_SIZE));
}
