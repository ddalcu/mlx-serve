//! Cold/explicit filesystem observations for the shared disk ledger.
const std = @import("std");
const Stat = std.Io.File.Stat;
const Inode = std.Io.File.INode;

pub const Observed = struct { path: []u8, stat: Stat };
pub const Inventory = struct {
    items: []Observed,
    pub fn deinit(self: *Inventory, a: std.mem.Allocator) void {
        for (self.items) |f| a.free(f.path);
        a.free(self.items);
    }
};
pub fn inventory(a: std.mem.Allocator, io: std.Io, path: []const u8) !Inventory {
    var files: std.ArrayList(Observed) = .empty;
    errdefer {
        for (files.items) |f| a.free(f.path);
        files.deinit(a);
    }
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return .{ .items = try files.toOwnedSlice(a) },
        else => return err,
    };
    defer dir.close(io);
    var walker = try dir.walkSelectively(a);
    defer walker.deinit();
    while (try walker.next(io)) |e| switch (e.kind) {
        .directory => walker.enter(io, e) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        },
        .file => {
            const st = e.dir.statFile(io, e.basename, .{ .follow_symlinks = false }) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            if (st.kind != .file) continue;
            const file_path = try std.fs.path.join(a, &.{ path, e.path });
            errdefer a.free(file_path);
            try files.append(a, .{ .path = file_path, .stat = st });
        },
        else => {},
    };
    return .{ .items = try files.toOwnedSlice(a) };
}

pub fn within(path: []const u8, root: []const u8) bool {
    const r = std.mem.trimEnd(u8, root, std.fs.path.sep_str);
    return std.mem.eql(u8, path, r) or (std.mem.startsWith(u8, path, r) and path.len > r.len and path[r.len] == std.fs.path.sep);
}
pub fn physicalBytes(a: std.mem.Allocator, io: std.Io, base: []const u8, exclude: ?[]const u8) !u64 {
    var observed = try inventory(a, io, base);
    defer observed.deinit(a);
    var seen: std.AutoHashMap(Inode, void) = .init(a);
    defer seen.deinit();
    var n: u64 = 0;
    for (observed.items) |f| {
        if (exclude) |path| if (within(f.path, path)) continue;
        if (!(try seen.getOrPut(f.stat.inode)).found_existing) n +|= f.stat.size;
    }
    return n;
}

/// Lexical translation only. Admission never resolves filesystem links.
pub fn canonicalTarget(a: std.mem.Allocator, root: []const u8, alias: []const u8, path: []const u8) ![]u8 {
    const resolved = try std.fs.path.resolve(a, &.{ root, path });
    if (within(resolved, root)) return resolved;
    defer a.free(resolved);
    if (!within(resolved, alias)) return error.BadDiskCacheDir;
    return std.fs.path.resolve(a, &.{ root, std.mem.trimStart(u8, resolved[alias.len..], std.fs.path.sep_str) });
}
