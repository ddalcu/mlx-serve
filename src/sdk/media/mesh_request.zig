//! A mesh request's photos: `image` (one view, every mesh backend) or `images` (2..max posed views, Pixal3D), each
//! decoded by the host from base64 into `Request.inputs` in body order and here to straight-alpha RGBA8. Each wrong
//! shape is refused by the field's name.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const image_pixels = @import("image_pixels.zig");

const Verdict = abi.Verdict;
const Obj = std.json.ObjectMap;

pub const BAD_IMAGES = "'images' must be an array of base64 PNG/JPEG views of one object";

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// A set field that is not JSON null.
pub fn has(obj: *const Obj, key: []const u8) bool {
    const v = obj.get(key) orelse return false;
    return v != .null;
}

/// Views decoded so far, freed together.
pub const Views = struct {
    list: std.ArrayList(image_pixels.Rgba) = .empty,

    pub fn deinit(self: *Views, a: std.mem.Allocator) void {
        for (self.list.items) |v| v.deinit(a);
        self.list.deinit(a);
    }

    /// One upload: `raw` is the body's string (empty is its own refusal), `bytes` what the host decoded from it.
    fn add(self: *Views, a: std.mem.Allocator, raw: []const u8, bytes: []const u8, comptime what: []const u8) ?Verdict {
        if (raw.len == 0) return bad("empty " ++ what);
        if (bytes.len == 0) return bad("invalid base64 in " ++ what);
        const img = image_pixels.decodeRgba(a, bytes) orelse return bad("could not decode " ++ what ++ " (PNG/JPEG supported)");
        self.list.append(a, img) catch {
            img.deinit(a);
            return Verdict.refuse(.failed, "out of memory");
        };
        return null;
    }
};

/// `image`, or with `max_views` > 1 `images` (2..max_views); a single-view backend refuses `images` by name.
pub fn decode(a: std.mem.Allocator, obj: *const Obj, inputs: []const abi.Str, comptime max_views: usize, out: *Views) ?Verdict {
    const input = struct {
        fn at(in: []const abi.Str, i: usize) []const u8 {
            return if (i < in.len) in[i].slice() else "";
        }
    }.at;
    if (!has(obj, "images")) {
        const v = obj.get("image") orelse return bad("missing 'image' (base64 PNG/JPEG of the subject)");
        if (v != .string) return bad("missing 'image' (base64 PNG/JPEG of the subject)");
        return out.add(a, v.string, input(inputs, 0), "'image'");
    }
    if (has(obj, "image")) return bad("send 'image' or 'images', not both");
    if (max_views < 2) return bad("'images' (several views of one object) is served by Pixal3D; this model takes one 'image'");
    const arr = obj.get("images").?;
    if (arr != .array) return bad(BAD_IMAGES);
    for (arr.array.items) |e| if (e != .string) return bad(BAD_IMAGES);
    const n = arr.array.items.len;
    if (n < 2) return bad("'images' needs at least 2 views (send a single view as 'image')");
    if (n > max_views) return bad(std.fmt.comptimePrint("'images' takes at most {d} views", .{max_views}));
    for (arr.array.items, 0..) |e, i| if (out.add(a, e.string, input(inputs, i), "an 'images' entry")) |v| return v;
    return null;
}

// ── tests ──

const testing = std.testing;

test "mesh request: image or images, each wrong shape refused by its field" {
    const a = testing.allocator;
    const png = @import("png.zig");
    const px = [_]u8{ 1, 2, 3, 255 };
    const one = try png.encodeRgba(a, &px, 1, 1);
    defer a.free(one);
    const Case = struct { body: []const u8, max: enum { one, two, eight }, want: ?[]const u8, n: usize = 0 };
    const cases = [_]Case{
        .{ .body = "{\"image\":\"x\"}", .max = .one, .want = null, .n = 1 },
        .{ .body = "{}", .max = .one, .want = "missing 'image' (base64 PNG/JPEG of the subject)" },
        .{ .body = "{\"image\":\"\"}", .max = .one, .want = "empty 'image'" },
        .{ .body = "{\"images\":[\"x\",\"y\"]}", .max = .one, .want = "'images' (several views of one object) is served by Pixal3D; this model takes one 'image'" },
        .{ .body = "{\"image\":\"x\",\"images\":[\"x\"]}", .max = .eight, .want = "send 'image' or 'images', not both" },
        .{ .body = "{\"images\":[1,2]}", .max = .eight, .want = BAD_IMAGES },
        .{ .body = "{\"images\":[\"x\"]}", .max = .eight, .want = "'images' needs at least 2 views (send a single view as 'image')" },
        .{ .body = "{\"images\":[\"x\",\"y\",\"z\"]}", .max = .two, .want = "'images' takes at most 2 views" },
        .{ .body = "{\"images\":[\"x\",\"y\",\"z\"]}", .max = .eight, .want = "invalid base64 in an 'images' entry", .n = 2 },
        .{ .body = "{\"images\":[\"x\",\"y\"]}", .max = .eight, .want = null, .n = 2 },
    };
    const inputs = [_]abi.Str{ .of(one), .of(one) };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        var vs: Views = .{};
        defer vs.deinit(a);
        const obj = &parsed.value.object;
        const v = switch (c.max) {
            .one => decode(a, obj, &inputs, 1, &vs),
            .two => decode(a, obj, &inputs, 2, &vs),
            .eight => decode(a, obj, &inputs, 8, &vs),
        };
        if (c.want) |w| try testing.expectEqualStrings(w, v.?.why.slice()) else try testing.expect(v == null);
        try testing.expectEqual(c.n, vs.list.items.len);
    }
}
