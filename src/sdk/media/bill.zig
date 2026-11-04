//! Media load bills from a pack's files, MLX-free (a plugin's `config.zig` runs them in discovery and admission).

const std = @import("std");
const common = @import("../../abi/common.zig");

const PackDir = common.PackDir;

/// The `.safetensors` bytes in folder `rel` ("" = the root) and in the folders directly under it; symlinked files
/// count (an HF hub snapshot is all symlinks).
pub fn safetensorsIn(dir: *const PackDir, rel: []const u8) u64 {
    return sumIn(dir, rel, 1);
}

fn sumIn(dir: *const PackDir, rel: []const u8, depth: u32) u64 {
    var buf: [16 * 1024]u8 = undefined;
    var it = dir.entries(rel, &buf) orelse return 0;
    var total: u64 = 0;
    while (it.next()) |name| {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = (if (rel.len == 0) std.fmt.bufPrint(&path_buf, "{s}", .{name}) else std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ rel, name })) catch continue;
        if (std.mem.endsWith(u8, name, "/")) {
            if (depth > 0) total += sumIn(dir, path[0 .. path.len - 1], depth - 1);
        } else if (std.mem.endsWith(u8, name, ".safetensors")) total += dir.fileBytes(path);
    }
    return total;
}

/// The summed sizes of `rels` (missing files count 0).
pub fn files(dir: *const PackDir, rels: []const []const u8) u64 {
    var total: u64 = 0;
    for (rels) |r| total += dir.fileBytes(r);
    return total;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../testing/fixture.zig");

test "media bill: the safetensors of a folder and the folders under it, nothing deeper or else" {
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "a.safetensors", .bytes = "1234" },
        .{ .name = "config.json", .bytes = "{}" },
        .{ .name = "te/model.safetensors", .bytes = "12" },
        .{ .name = "te/deep/x.safetensors", .bytes = "123456789" },
    } };
    const dir = mem.packDir();
    try testing.expectEqual(@as(u64, 6), safetensorsIn(&dir, ""));
    try testing.expectEqual(@as(u64, 11), safetensorsIn(&dir, "te"));
    try testing.expectEqual(@as(u64, 6), files(&dir, &.{ "a.safetensors", "te/model.safetensors", "missing" }));
}
