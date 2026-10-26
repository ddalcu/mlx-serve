//! Image bytes to and from MLX for the image backends: an upload's native size, an upload decoded to a cover-cropped
//! [1, 3, H, W] f32 in [0, 1] (or to RGBA8 for the mesh backends), and a finished [1, C, H, W] image (C = 3 or 4)
//! encoded as PNG.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const stb = @import("stb");
const png = @import("png.zig");
const log = @import("../util/log.zig");

pub const Size = struct { w: u32, h: u32 };

/// An encoded PNG/JPEG's pixel size, without decoding its pixels; null when stb cannot read it.
pub fn nativeSize(encoded: []const u8) ?Size {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    if (stb.stbi_info_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch) == 0) return null;
    if (w <= 0 or h <= 0) return null;
    return .{ .w = @intCast(w), .h = @intCast(h) };
}

/// Straight-alpha RGBA8 pixels at the image's own size (a 3-channel source gets opaque alpha).
pub const Rgba = struct {
    pix: []u8,
    w: u32,
    h: u32,

    pub fn deinit(self: Rgba, allocator: std.mem.Allocator) void {
        allocator.free(self.pix);
    }
};

/// PNG/JPEG bytes → owned RGBA8 (the mesh backends' input); null when the bytes do not decode.
pub fn decodeRgba(allocator: std.mem.Allocator, encoded: []const u8) ?Rgba {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 4) orelse return null;
    defer stb.stbi_image_free(src);
    if (w <= 0 or h <= 0) return null;
    const n: usize = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 4;
    const pix = allocator.dupe(u8, src[0..n]) catch return null;
    return .{ .pix = pix, .w = @intCast(w), .h = @intCast(h) };
}

/// The engagement lines of an edit whose backend takes the encoded references as they are (refs[0] = the primary).
pub fn logByteRefs(refs: []const []const u8) void {
    for (refs, 1..) |r, i| if (i == 1)
        log.info("[image] edit: reference {d} bytes (byte-based backend)\n", .{r.len})
    else
        log.info("[image] edit ref {d}: {d} bytes (byte-based backend)\n", .{ i, r.len });
}

/// An img2img request's source (`decodeCover`), with the engagement line its tests read.
pub fn decodeVariation(allocator: std.mem.Allocator, encoded: []const u8, target_h: u32, target_w: u32, strength: f32) ?mlx.mlx_array {
    const pix = decodeCover(allocator, encoded, target_h, target_w) orelse return null;
    log.info("[image] img2img: source {d} bytes, strength={d:.2}\n", .{ encoded.len, strength });
    return pix;
}

/// PNG/JPEG bytes → [1, 3, target_h, target_w] f32 in [0, 1] (evaluated; caller frees), bilinear over the largest
/// centred source window of the target's aspect: never stretched, a mismatched aspect loses its edges. Null when
/// the bytes do not decode.
pub fn decodeCover(allocator: std.mem.Allocator, encoded: []const u8, target_h: u32, target_w: u32) ?mlx.mlx_array {
    var w: c_int = 0;
    var h: c_int = 0;
    var ch: c_int = 0;
    const src_ptr = stb.stbi_load_from_memory(encoded.ptr, @intCast(encoded.len), &w, &h, &ch, 3) orelse return null;
    defer stb.stbi_image_free(src_ptr);
    const sw: usize = @intCast(w);
    const sh: usize = @intCast(h);
    if (sw == 0 or sh == 0) return null;
    const src = src_ptr[0 .. sw * sh * 3];

    const th: usize = target_h;
    const tw: usize = target_w;
    const out = allocator.alloc(f32, 3 * th * tw) catch return null;
    defer allocator.free(out);

    const src_ar = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(sh));
    const tgt_ar = @as(f32, @floatFromInt(tw)) / @as(f32, @floatFromInt(th));
    var win_w: f32 = @floatFromInt(sw);
    var win_h: f32 = @floatFromInt(sh);
    if (src_ar > tgt_ar) win_w = win_h * tgt_ar else win_h = win_w / tgt_ar;
    const x_off = (@as(f32, @floatFromInt(sw)) - win_w) * 0.5;
    const y_off = (@as(f32, @floatFromInt(sh)) - win_h) * 0.5;

    const clampIdx = struct {
        fn f(v: isize, n: usize) usize {
            if (v < 0) return 0;
            const uv: usize = @intCast(v);
            return if (uv >= n) n - 1 else uv;
        }
    }.f;

    for (0..th) |oy| {
        const fy = y_off + (@as(f32, @floatFromInt(oy)) + 0.5) * win_h / @as(f32, @floatFromInt(th)) - 0.5;
        const fy0 = @floor(fy);
        const wy = fy - fy0;
        const y0 = clampIdx(@intFromFloat(fy0), sh);
        const y1 = clampIdx(@as(isize, @intFromFloat(fy0)) + 1, sh);
        for (0..tw) |ox| {
            const fx = x_off + (@as(f32, @floatFromInt(ox)) + 0.5) * win_w / @as(f32, @floatFromInt(tw)) - 0.5;
            const fx0 = @floor(fx);
            const wx = fx - fx0;
            const x0 = clampIdx(@intFromFloat(fx0), sw);
            const x1 = clampIdx(@as(isize, @intFromFloat(fx0)) + 1, sw);
            for (0..3) |c| {
                const p00: f32 = @floatFromInt(src[(y0 * sw + x0) * 3 + c]);
                const p10: f32 = @floatFromInt(src[(y0 * sw + x1) * 3 + c]);
                const p01: f32 = @floatFromInt(src[(y1 * sw + x0) * 3 + c]);
                const p11: f32 = @floatFromInt(src[(y1 * sw + x1) * 3 + c]);
                const top = p00 * (1.0 - wx) + p10 * wx;
                const bot = p01 * (1.0 - wx) + p11 * wx;
                out[c * th * tw + oy * tw + ox] = (top * (1.0 - wy) + bot * wy) / 255.0;
            }
        }
    }
    const arr = mlx.mlx_array_new_data(out.ptr, &[_]c_int{ 1, 3, @intCast(th), @intCast(tw) }, 4, .float32);
    _ = mlx.mlx_array_eval(arr);
    return arr;
}

/// [1, 3|4, H, W] f32 in [0, 1] → PNG bytes (RGB, or RGBA for 4 channels; caller frees). Evaluates the image: a
/// lazy generation's failure surfaces here.
pub fn toPng(allocator: std.mem.Allocator, img: mlx.mlx_array, s: mlx.mlx_stream) ![]u8 {
    var cf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf);
    try mlx.check(mlx.mlx_contiguous(&cf, img, false, s));
    try mlx.check(mlx.mlx_array_eval(cf));
    const sh = mlx.getShape(cf);
    if (sh.len != 4 or sh[0] != 1 or (sh[1] != 3 and sh[1] != 4)) return error.InvalidImageShape;
    const channels: usize = @intCast(sh[1]);
    const H: usize = @intCast(sh[2]);
    const W: usize = @intCast(sh[3]);
    const d = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const pixels = try allocator.alloc(u8, W * H * channels);
    defer allocator.free(pixels);
    const plane = W * H;
    for (0..H) |y| for (0..W) |x| {
        const o = (y * W + x) * channels;
        for (0..channels) |c| pixels[o + c] = @intFromFloat(std.math.clamp(d[c * plane + y * W + x] * 255.0, 0, 255));
    };
    return if (channels == 4) png.encodeRgba(allocator, pixels, @intCast(W), @intCast(H)) else png.encodeRgb(allocator, pixels, @intCast(W), @intCast(H));
}

// ── tests ──

const testing = std.testing;

test "image pixels: a PNG round-trips through decode and encode at its own size" {
    const a = testing.allocator;
    const rgb = [_]u8{ 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };
    const bytes = try png.encodeRgb(a, &rgb, 2, 2);
    defer a.free(bytes);
    try testing.expectEqual(Size{ .w = 2, .h = 2 }, nativeSize(bytes).?);
    try testing.expect(nativeSize("not an image") == null);
    const img = decodeCover(a, bytes, 2, 2).?;
    defer _ = mlx.mlx_array_free(img);
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const again = try toPng(a, img, s);
    defer a.free(again);
    try testing.expectEqualSlices(u8, bytes, again);
    const rgba = decodeRgba(a, bytes).?;
    defer rgba.deinit(a);
    try testing.expect(rgba.w == 2 and rgba.h == 2);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255, 0, 255, 0, 255 }, rgba.pix[0..8]);
    try testing.expect(decodeRgba(a, "not an image") == null);
}
