//! The template media backend's claims, caps and bills: MLX-free (discovery and admission run them with no device);
//! the build checks this file's import closure.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");

pub const model_types = [_][]const u8{"_template_media"};

pub const claims = sdk.mediaClaimsByType(&model_types);

pub const max_dim = 64;
pub const max_steps = 50;

/// A job's body, bounds-checked (request JSON is untrusted).
pub const Body = struct {
    size: u32 = 8,
    steps: u32 = 2,
    seed: u64 = 0,

    pub fn parse(json: []const u8) ?Body {
        var buf: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const b = std.json.parseFromSliceLeaky(Body, fba.allocator(), json, .{ .ignore_unknown_fields = true }) catch return null;
        if (b.size == 0 or b.size > max_dim or b.steps == 0 or b.steps > max_steps) return null;
        return b;
    }
};

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .max_dim = max_dim };
}

/// The weights' bytes, from the safetensors header's largest data offset.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    var buf: [1 << 16]u8 = undefined;
    var len: usize = 0;
    if (!dir.safetensors_header(dir.ctx, .of("model.safetensors"), &buf, buf.len, &len).isOk()) return 0;
    var fba_buf: [1 << 16]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const v = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), buf[0..len], .{}) catch return 0;
    if (v != .object) return 0;
    var end: u64 = 0;
    for (v.object.values()) |t| {
        if (t != .object) continue;
        const offs = t.object.get("data_offsets") orelse continue;
        if (offs != .array or offs.array.items.len != 2 or offs.array.items[1] != .integer) continue;
        end = @max(end, @as(u64, @intCast(@max(offs.array.items[1].integer, 0))));
    }
    return end;
}

/// Three f32 planes of size x size x 4 live at once (noise, its update, the colour projection's input).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    const b = Body.parse(body.slice()) orelse return 0;
    return @as(u64, b.size) * b.size * 4 * 4 * 3;
}

// ── tests ──

const testing = std.testing;

test "_template_media config: a body out of bounds is refused, a good one bills by its size" {
    try testing.expect(Body.parse("{\"size\":0}") == null and Body.parse("{\"size\":65}") == null and Body.parse("{\"steps\":51}") == null);
    try testing.expect(Body.parse("{") == null);
    const dir: abi.PackDir = undefined;
    try testing.expectEqual(@as(u64, 4 * 4 * 48), billRequest(&dir, &.{}, .of("{\"size\":4}")));
}
