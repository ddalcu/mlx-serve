//! A video result as the `/v1/video/generations` response carries it: raw RGB frames and optional PCM16 audio, both
//! base64, as JSON members (`"frames":…,"data":"…"`; the host wraps them). MLX-free.

const std = @import("std");

/// Ceiling on one response's raw RGB volume: the whole buffer is base64'd into ONE JSON body the app decodes in
/// memory, and past this the client drops the socket and a finished render dies as `WriteFailed` (#283). Twin of the
/// app's `VideoModelPreset.maxFramePayloadBytes`; refused at admission, by name, before any denoise step.
pub const MAX_RGB_BYTES: u64 = 768 * 1024 * 1024;

/// The refusal for `rgbBytes` past the cap: delivered frames, width, height, MB, cap MB.
pub const TOO_LARGE_FMT = "{d} frames at {d}x{d} is {d} MB of raw RGB; one response carries at most {d} MB — fewer frames or windows, or a smaller canvas";
pub const TOO_LARGE = "video too large for one response";

pub fn rgbBytes(frames: u32, width: u32, height: u32) u64 {
    return @as(u64, frames) * @as(u64, width) * @as(u64, height) * 3;
}

pub const Audio = struct { sample_rate: u32, channels: u32, pcm16: []const u8 };

/// The members for `rgb` ([frames, height, width, 3] u8) and its soundtrack. Caller frees.
pub fn members(a: std.mem.Allocator, frames: u32, height: u32, width: u32, fps: u32, rgb: []const u8, audio: ?Audio) ![]u8 {
    const enc = std.base64.standard.Encoder;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.ensureTotalCapacity(a, enc.calcSize(rgb.len) + (if (audio) |x| enc.calcSize(x.pcm16.len) else 0) + 256);
    try out.print(a, "\"frames\":{d},\"height\":{d},\"width\":{d},\"fps\":{d},\"format\":\"rgb8\",\"data\":\"", .{ frames, height, width, fps });
    try appendB64(a, &out, rgb);
    try out.append(a, '"');
    if (audio) |x| {
        try out.print(a, ",\"audio_sample_rate\":{d},\"audio_channels\":{d},\"audio_format\":\"pcm_s16le\",\"audio_data\":\"", .{ x.sample_rate, x.channels });
        try appendB64(a, &out, x.pcm16);
        try out.append(a, '"');
    }
    return out.toOwnedSlice(a);
}

/// f32 samples in [-1, 1] (clamped) as PCM16 little-endian bytes. Caller frees.
pub fn pcm16le(a: std.mem.Allocator, pcm: []const f32) ![]u8 {
    const out = try a.alloc(u8, pcm.len * 2);
    for (pcm, 0..) |v, i| {
        const iv: i16 = @intFromFloat(@round(@max(@as(f32, -1.0), @min(@as(f32, 1.0), v)) * 32767.0));
        std.mem.writeInt(i16, out[i * 2 ..][0..2], iv, .little);
    }
    return out;
}

fn appendB64(a: std.mem.Allocator, out: *std.ArrayList(u8), bytes: []const u8) !void {
    const enc = std.base64.standard.Encoder;
    const n = enc.calcSize(bytes.len);
    try out.ensureUnusedCapacity(a, n);
    _ = enc.encode(out.unusedCapacitySlice()[0..n], bytes);
    out.items.len += n;
}

// ── tests ──

const testing = std.testing;

test "video out: the members carry the frames, then the soundtrack, base64" {
    const m = try members(testing.allocator, 1, 1, 1, 24, &.{ 1, 2, 3 }, .{ .sample_rate = 32000, .channels = 2, .pcm16 = &.{ 0, 0 } });
    defer testing.allocator.free(m);
    try testing.expectEqualStrings("\"frames\":1,\"height\":1,\"width\":1,\"fps\":24,\"format\":\"rgb8\",\"data\":\"AQID\",\"audio_sample_rate\":32000,\"audio_channels\":2,\"audio_format\":\"pcm_s16le\",\"audio_data\":\"AAA=\"", m);
    try testing.expect(rgbBytes(701, 1056, 864) > MAX_RGB_BYTES);
}

test "pcm16le converts, clamps, and is little-endian" {
    const alloc = testing.allocator;
    // 0.0 → 0; 1.0 → 32767; -1.0 → -32767; out-of-range clamps; midscale rounds.
    const pcm = [_]f32{ 0.0, 1.0, -1.0, 2.0, -2.0, 0.5 };
    const bytes = try pcm16le(alloc, &pcm);
    defer alloc.free(bytes);
    try testing.expectEqual(@as(usize, 12), bytes.len);
    const read = struct {
        fn le(b: []const u8, i: usize) i16 {
            return @bitCast(@as(u16, b[i * 2]) | (@as(u16, b[i * 2 + 1]) << 8));
        }
    }.le;
    try testing.expectEqual(@as(i16, 0), read(bytes, 0));
    try testing.expectEqual(@as(i16, 32767), read(bytes, 1));
    try testing.expectEqual(@as(i16, -32767), read(bytes, 2));
    try testing.expectEqual(@as(i16, 32767), read(bytes, 3)); // 2.0 clamps to 1.0
    try testing.expectEqual(@as(i16, -32767), read(bytes, 4)); // -2.0 clamps to -1.0
    try testing.expectEqual(@as(i16, @intFromFloat(@round(0.5 * 32767.0))), read(bytes, 5));
}
