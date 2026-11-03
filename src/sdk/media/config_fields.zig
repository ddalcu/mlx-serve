//! A pack's config.json numbers into a geometry struct, MLX-free: each key present overrides its field's default
//! (the published values), so a tiny fixture pack states only what it shrinks.

const std = @import("std");

pub const Error = error{BadPackConfig};

/// `keys` is a tuple of `.{ json_key, field_name }`; integers fill integer fields (positive), numbers float ones.
pub fn fill(comptime T: type, comptime keys: anytype, obj: std.json.ObjectMap, out: *T) Error!void {
    inline for (keys) |k| if (obj.get(k[0])) |v| {
        const F = @TypeOf(@field(out.*, k[1]));
        @field(out.*, k[1]) = switch (F) {
            u32, i32, c_int => if (v == .integer and v.integer > 0 and v.integer <= std.math.maxInt(F)) @intCast(v.integer) else return error.BadPackConfig,
            f32, f64 => switch (v) {
                .integer => |i| @floatFromInt(i),
                .float => |f| @floatCast(f),
                else => return error.BadPackConfig,
            },
            else => @compileError("config_fields.fill: " ++ @typeName(F)),
        };
    };
}

/// The object `name` inside `obj`, or an empty one when absent.
pub fn section(obj: std.json.ObjectMap, name: []const u8) Error!std.json.ObjectMap {
    const v = obj.get(name) orelse return .empty;
    return if (v == .object) v.object else error.BadPackConfig;
}

/// `model_dir/rel` read into `buf` (configs are small; one that fills `buf` is refused).
pub fn read(io: std.Io, buf: []u8, model_dir: []const u8, rel: []const u8) ![]u8 {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, rel });
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var r = f.reader(io, &.{});
    const n = try r.interface.readSliceShort(buf);
    if (n == buf.len) return error.BadPackConfig;
    return buf[0..n];
}

test "config fields: present keys override, absent keep the default, a wrong type is refused" {
    const G = struct { width: u32 = 8, eps: f32 = 1e-6 };
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"w\":4,\"inner\":{\"e\":1}}", .{});
    defer parsed.deinit();
    var g = G{};
    try fill(G, .{.{ "w", "width" }}, parsed.value.object, &g);
    try fill(G, .{.{ "e", "eps" }}, try section(parsed.value.object, "inner"), &g);
    try std.testing.expectEqualDeep(G{ .width = 4, .eps = 1 }, g);
    try std.testing.expectError(error.BadPackConfig, fill(G, .{.{ "inner", "width" }}, parsed.value.object, &g));
}
