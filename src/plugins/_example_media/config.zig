//! The example media backend's claims, caps and bills: MLX-free (discovery and admission run them with no device).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.media;

pub const model_types = [_][]const u8{"_example_media"};

pub const claims = kit.sdk.mediaClaimsByType(&model_types);

pub const sample_rate = 16000;
pub const max_seconds = 30;
pub const chunks = 4;

/// A job's body, bounds-checked (request JSON is untrusted).
pub const Body = struct {
    seconds: f32 = 1,
    frequency: f32 = 440,

    pub fn parse(json: []const u8) ?Body {
        var buf: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const b = std.json.parseFromSliceLeaky(Body, fba.allocator(), json, .{ .ignore_unknown_fields = true }) catch return null;
        if (!(b.seconds > 0 and b.seconds <= max_seconds) or !(b.frequency > 0 and b.frequency < sample_rate / 2)) return null;
        return b;
    }
    pub fn samples(b: Body) u32 {
        return @intFromFloat(@ceil(b.seconds * sample_rate));
    }
};

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .sound = true } };
}

/// The weights as stored.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return dir.fileBytes("model.safetensors");
}

/// The f32 tone and its 16-bit copy.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    const b = Body.parse(body.slice()) orelse return 0;
    return @as(u64, b.samples()) * 6;
}

// ── tests ──

const testing = std.testing;

test "_example_media config: a body out of bounds is refused; a good one bills by its length" {
    try testing.expect(Body.parse("{\"seconds\":0}") == null and Body.parse("{\"seconds\":31}") == null and Body.parse("{\"frequency\":9000}") == null and Body.parse("{") == null);
    const dir: abi.PackDir = undefined;
    try testing.expectEqual(@as(u64, 8000 * 6), billRequest(&dir, &.{}, .of("{\"seconds\":0.5}")));
}
