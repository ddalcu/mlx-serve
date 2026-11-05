//! Shared KV disk headroom. Other namespaces are measured read-only; each engine owns its eviction.

const std = @import("std");
const testing = std.testing;

/// Growth headroom under one finite root-wide cap. Pass zero `own_logical_used` for the total
/// allowance of `own_root`; null means no room, never an unbounded zero. Files in that namespace
/// are replaced by the caller's logical bill, which may include writes not yet on disk.
pub fn availableBudget(allocator: std.mem.Allocator, io: std.Io, base: []const u8, own_root: []const u8, cap: u64, own_logical_used: u64) !?u64 {
    if (own_logical_used >= cap) return null;
    const other = try physicalBytes(allocator, io, base, own_root);
    const available = cap -| other -| own_logical_used;
    return if (available == 0) null else available;
}

/// Actual regular-file sizes, including temporary writes; optionally exclude one owned subtree.
/// The common cache root lives on one volume; inherited hard links are billed once by inode.
/// Symlinks are not followed, and files disappearing during another engine's eviction are skipped.
pub fn physicalBytes(allocator: std.mem.Allocator, io: std.Io, base: []const u8, exclude: ?[]const u8) !u64 {
    if (!std.fs.path.isAbsolute(base)) return error.BadDiskCacheDir;
    const own = if (exclude) |p| blk: {
        if (!std.fs.path.isAbsolute(p)) return error.BadDiskCacheDir;
        break :blk try std.fs.path.relativeAlloc(allocator, base, null, base, p);
    } else null;
    defer if (own) |p| allocator.free(p);
    if (own) |p| {
        if (p.len == 0) return 0;
        if (std.fs.path.isAbsolute(p) or std.mem.eql(u8, p, "..") or std.mem.startsWith(u8, p, ".." ++ std.fs.path.sep_str)) return error.BadDiskCacheDir;
    }
    var dir = std.Io.Dir.openDirAbsolute(io, base, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer dir.close(io);
    var walker = try dir.walkSelectively(allocator);
    defer walker.deinit();
    var linked: std.AutoHashMap(std.Io.File.INode, void) = .init(allocator);
    defer linked.deinit();
    var total: u64 = 0;
    while (try walker.next(io)) |entry| {
        switch (entry.kind) {
            .directory => {
                if (own) |p| if (std.mem.eql(u8, entry.path, p)) continue;
                walker.enter(io, entry) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
            },
            .file => {
                const st = entry.dir.statFile(io, entry.basename, .{ .follow_symlinks = false }) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                if (st.kind != .file) continue;
                if (st.nlink > 1 and (try linked.getOrPut(st.inode)).found_existing) continue;
                total +|= st.size;
            },
            else => {},
        }
    }
    return total;
}

test "shared disk budget counts MLX ds4 and unknown namespaces against one cap" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx/e1");
    try tmp.dir.createDirPath(io, "ds4/model/kv");
    try tmp.dir.createDirPath(io, "another-engine/state");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/e1/chunk", .data = &@as([30]u8, @splat(1)) });
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/model/kv/save.tmp", .data = &@as([40]u8, @splat(2)) });
    try tmp.dir.writeFile(io, .{ .sub_path = "another-engine/state/blob", .data = &@as([10]u8, @splat(3)) });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const mlx_root = try std.fs.path.join(a, &.{ base, "mlx" });
    defer a.free(mlx_root);
    const ds4_root = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(ds4_root);
    try testing.expectEqual(@as(?u64, 50), try availableBudget(a, io, base, mlx_root, 100, 0));
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, mlx_root, 100, 30));
    try testing.expectEqual(@as(?u64, 60), try availableBudget(a, io, base, ds4_root, 100, 0));
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, ds4_root, 100, 40));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 50, 0));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 100, 50));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 0, 0));
    // A native save being replaced is measured by its current size, without interpreting metadata.
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/model/kv/save.tmp", .data = &@as([70]u8, @splat(4)) });
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, mlx_root, 100, 0));
    try testing.expectEqual(@as(u64, 70), (try tmp.dir.statFile(io, "ds4/model/kv/save.tmp", .{})).size);
}

test "shared disk budget excludes only the exact own subtree and counts hard links once" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "native/own");
    try tmp.dir.createDirPath(io, "native/own-sibling");
    try tmp.dir.writeFile(io, .{ .sub_path = "native/own/ignored", .data = "ignored own data" });
    try tmp.dir.writeFile(io, .{ .sub_path = "native/own-sibling/blob", .data = "1234567" });
    try std.Io.Dir.hardLink(tmp.dir, "native/own-sibling/blob", tmp.dir, "linked", io, .{});
    try tmp.dir.symLink(io, "native/own-sibling", "symlink", .{ .is_directory = true });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const own = try std.fs.path.join(a, &.{ base, "native", "own", "." });
    defer a.free(own);
    try testing.expectEqual(@as(?u64, 93), try availableBudget(a, io, base, own, 100, 0));
}

test "shared disk budget follows the common-root SSD symlink at startup" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "ssd/kv-cache/ds4");
    try tmp.dir.writeFile(io, .{ .sub_path = "ssd/kv-cache/ds4/state.kv", .data = "native" });
    try tmp.dir.symLink(io, "ssd/kv-cache", "kv-cache", .{ .is_directory = true });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const home = buf[0..try tmp.dir.realPath(io, &buf)];
    const base = try std.fs.path.join(a, &.{ home, "kv-cache" });
    defer a.free(base);
    const own = try std.fs.path.join(a, &.{ base, "0000000000000001" });
    defer a.free(own);
    // The model's namespace need not exist before its startup allowance is measured.
    try testing.expectEqual(@as(?u64, 94), try availableBudget(a, io, base, own, 100, 0));
    try tmp.dir.createDirPath(io, "ssd/kv-cache/0000000000000001");
    try tmp.dir.writeFile(io, .{ .sub_path = "ssd/kv-cache/0000000000000001/chunk", .data = "owned" });
    try testing.expectEqual(@as(?u64, 94), try availableBudget(a, io, base, own, 100, 0));
    try testing.expectEqual(@as(u64, 11), try physicalBytes(a, io, base, null));
}
