//! Shared KV disk headroom. Other namespaces are measured read-only; each engine owns its eviction.

const std = @import("std");
const testing = std.testing;
const Shared = @import("../util/shared.zig").Shared;

const Reservations = struct {
    mutex: std.Io.Mutex = .init,
    first: ?*Reservation = null,
    generation: u64 = 0,
};
pub const reservations_shared = Shared(Reservations, "kv_disk_reservations", .{});

/// An existing export's byte promise, covering one directory until its writes settle.
/// Paths are owned: neither a moved tier nor an asynchronous writer is borrowed here.
pub const Reservation = struct {
    allocator: std.mem.Allocator,
    base: []u8,
    root: []u8,
    bytes: u64,
    next: ?*Reservation = null,

    /// Cancellation retains only already-accepted writes until their cleanup finishes.
    pub fn shrink(self: *Reservation, io: std.Io, bytes: u64) void {
        const state = reservations_shared.get();
        state.mutex.lockUncancelable(io);
        defer state.mutex.unlock(io);
        std.debug.assert(bytes <= self.bytes);
        self.bytes = bytes;
    }

    pub fn release(self: *Reservation, io: std.Io) void {
        const state = reservations_shared.get();
        state.mutex.lockUncancelable(io);
        var link = &state.first;
        while (link.* != self) link = &link.*.?.next;
        link.* = self.next;
        state.mutex.unlock(io);
        self.allocator.free(self.base);
        self.allocator.free(self.root);
        self.allocator.destroy(self);
    }
};

const Claim = struct { relative: []u8, bytes: u64, visible: u64 = 0 };
const Snapshot = struct {
    claims: std.ArrayList(Claim) = .empty,
    generation: u64,

    fn deinit(self: *Snapshot, allocator: std.mem.Allocator) void {
        for (self.claims.items) |c| allocator.free(c.relative);
        self.claims.deinit(allocator);
    }
};

fn within(path: []const u8, root: []const u8) bool {
    return std.mem.eql(u8, path, root) or (std.mem.startsWith(u8, path, root) and
        path.len > root.len and path[root.len] == std.fs.path.sep);
}

fn snapshot(allocator: std.mem.Allocator, io: std.Io, base: []const u8, own: []const u8) !Snapshot {
    const state = reservations_shared.get();
    state.mutex.lockUncancelable(io);
    defer state.mutex.unlock(io);
    var result: Snapshot = .{ .generation = state.generation };
    errdefer result.deinit(allocator);
    var next = state.first;
    while (next) |r| : (next = r.next) {
        if (!std.mem.eql(u8, r.base, base) or within(r.root, own)) continue;
        const relative = try std.fs.path.relativeAlloc(allocator, base, null, base, r.root);
        errdefer allocator.free(relative);
        try result.claims.append(allocator, .{ .relative = relative, .bytes = r.bytes });
    }
    return result;
}

fn paths(allocator: std.mem.Allocator, io: std.Io, base: []const u8, own: []const u8) !struct { base: []u8, root: []u8 } {
    if (!std.fs.path.isAbsolute(base) or !std.fs.path.isAbsolute(own)) return error.BadDiskCacheDir;
    const relative = try std.fs.path.relativeAlloc(allocator, base, null, base, own);
    defer allocator.free(relative);
    if (std.fs.path.isAbsolute(relative) or std.mem.eql(u8, relative, "..") or std.mem.startsWith(u8, relative, ".." ++ std.fs.path.sep_str)) return error.BadDiskCacheDir;
    const resolved = blk: {
        var buffer: [std.fs.max_path_bytes]u8 = undefined;
        const n = std.Io.Dir.cwd().realPathFile(io, base, &buffer) catch |err| switch (err) {
            error.FileNotFound => break :blk try std.fs.path.resolve(allocator, &.{base}),
            else => return err,
        };
        break :blk try allocator.dupe(u8, buffer[0..n]);
    };
    errdefer allocator.free(resolved);
    return .{ .base = resolved, .root = try std.fs.path.resolve(allocator, &.{ resolved, relative }) };
}

/// Reserve an exact module bill, or (bytes=null) the native pool's whole allowance.
/// A concurrent admission declines this attempt instead of waiting or retrying.
/// Releases/shrinks leave snapshots conservative and do not invalidate a producer's scan.
/// No lock spans filesystem I/O, GPU readback, or the writer's work.
pub fn reserve(allocator: std.mem.Allocator, io: std.Io, base: []const u8, own_root: []const u8, cap: u64, bytes: ?u64) !?*Reservation {
    const p = try paths(allocator, io, base, own_root);
    var owned = true;
    defer if (owned) {
        allocator.free(p.base);
        allocator.free(p.root);
    };
    var snap = try snapshot(allocator, io, p.base, p.root);
    defer snap.deinit(allocator);
    const used = try measure(allocator, io, p.base, p.root, snap.claims.items);
    const available = cap -| used;
    const bill = bytes orelse available;
    if (bill == 0 or bill > available) return null;
    const r = try allocator.create(Reservation);
    errdefer allocator.destroy(r);
    r.* = .{ .allocator = allocator, .base = p.base, .root = p.root, .bytes = bill };
    const state = reservations_shared.get();
    state.mutex.lockUncancelable(io);
    defer state.mutex.unlock(io);
    if (snap.generation != state.generation) {
        allocator.destroy(r);
        return null;
    }
    var next = state.first;
    while (next) |other| : (next = other.next) {
        if (within(other.root, p.root) or within(p.root, other.root)) {
            allocator.destroy(r);
            return null;
        }
    }
    r.next = state.first;
    state.first = r;
    state.generation +%= 1;
    owned = false;
    return r;
}

/// Growth headroom under one finite root-wide cap. Pass zero `own_logical_used` for the total
/// allowance of `own_root`; null means no room, never an unbounded zero. Files in that namespace
/// are replaced by the caller's logical bill, which may include writes not yet on disk.
pub fn availableBudget(allocator: std.mem.Allocator, io: std.Io, base: []const u8, own_root: []const u8, cap: u64, own_logical_used: u64) !?u64 {
    if (own_logical_used >= cap) return null;
    const p = try paths(allocator, io, base, own_root);
    defer allocator.free(p.base);
    defer allocator.free(p.root);
    var snap = try snapshot(allocator, io, p.base, p.root);
    defer snap.deinit(allocator);
    const other = try measure(allocator, io, p.base, p.root, snap.claims.items);
    const state = reservations_shared.get();
    state.mutex.lockUncancelable(io);
    defer state.mutex.unlock(io);
    if (snap.generation != state.generation) return null;
    const available = cap -| other -| own_logical_used;
    return if (available == 0) null else available;
}

/// Actual regular-file sizes, including temporary writes; optionally exclude one owned subtree.
/// The common cache root lives on one volume; inherited hard links are billed once by inode.
/// Symlinks are not followed, and files disappearing during another engine's eviction are skipped.
pub fn physicalBytes(allocator: std.mem.Allocator, io: std.Io, base: []const u8, exclude: ?[]const u8) !u64 {
    return measure(allocator, io, base, exclude, &.{});
}

fn measure(allocator: std.mem.Allocator, io: std.Io, base: []const u8, exclude: ?[]const u8, claims: []Claim) !u64 {
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
        error.FileNotFound => {
            var reserved: u64 = 0;
            for (claims) |c| reserved +|= c.bytes;
            return reserved;
        },
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
                if (own) |p| if (std.mem.eql(u8, entry.path, p)) continue;
                const st = entry.dir.statFile(io, entry.basename, .{ .follow_symlinks = false }) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                if (st.kind != .file) continue;
                if (st.nlink > 1 and (try linked.getOrPut(st.inode)).found_existing) continue;
                total +|= st.size;
                // Measure both terms from this SAME stat. Independent scans could see
                // growth between them and subtract bytes never added to the physical bill.
                for (claims) |*c| if (within(entry.path, c.relative)) {
                    c.visible +|= st.size;
                };
            },
            else => {},
        }
    }
    for (claims) |c| total +|= c.bytes -| c.visible;
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

test "shared disk reservations charge pending bytes once while files become visible" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx/e1");
    try tmp.dir.createDirPath(io, "ds4");
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const module = try std.fs.path.join(a, &.{ base, "mlx/e1" });
    defer a.free(module);
    const native = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(native);
    const reservation = (try reserve(a, io, base, module, 100, 80)).?;
    defer reservation.release(io);
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, native, 100, 0));
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/e1/part.tmp", .data = &@as([30]u8, @splat(1)) });
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, native, 100, 0));
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/e1/part.tmp", .data = &@as([80]u8, @splat(1)) });
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, native, 100, 0));
    try tmp.dir.writeFile(io, .{ .sub_path = "unindexed", .data = "unknown" });
    try testing.expectEqual(@as(?u64, 13), try availableBudget(a, io, base, native, 100, 0));
    // Cancellation releases unwritten bytes, but visible remnants still occupy the cap.
    reservation.shrink(io, 0);
    try testing.expectEqual(@as(?u64, 13), try availableBudget(a, io, base, native, 100, 0));
    try tmp.dir.deleteFile(io, "mlx/e1/part.tmp");
    try testing.expectEqual(@as(?u64, 93), try availableBudget(a, io, base, native, 100, 0));
}

test "shared disk reservations cannot concurrently admit two oversized exports" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const roots = [_][]u8{
        try std.fs.path.join(a, &.{ base, "mlx" }),
        try std.fs.path.join(a, &.{ base, "ds4" }),
    };
    defer for (roots) |root| a.free(root);
    const Worker = struct {
        fn run(base_path: []const u8, root: []const u8, out: *?*Reservation, err: *?anyerror) void {
            out.* = reserve(testing.allocator, testing.io, base_path, root, 100, 80) catch |e| {
                err.* = e;
                return;
            };
        }
    };
    var admitted: [2]?*Reservation = @splat(null);
    defer for (admitted) |r| if (r) |p| p.release(io);
    var errors: [2]?anyerror = @splat(null);
    const t1 = try std.Thread.spawn(.{}, Worker.run, .{ base, roots[0], &admitted[0], &errors[0] });
    const t2 = std.Thread.spawn(.{}, Worker.run, .{ base, roots[1], &admitted[1], &errors[1] }) catch |err| {
        t1.join();
        return err;
    };
    t1.join();
    t2.join();
    for (errors) |err| if (err) |e| return e;
    try testing.expect((admitted[0] != null) != (admitted[1] != null));
}

test "shared disk reservations follow SSD root aliases without charging another root" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "ssd/mlx/e1");
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.symLink(io, "ssd", "kv-cache", .{ .is_directory = true });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const home = buf[0..try tmp.dir.realPath(io, &buf)];
    const alias = try std.fs.path.join(a, &.{ home, "kv-cache" });
    defer a.free(alias);
    const direct = try std.fs.path.join(a, &.{ home, "ssd" });
    defer a.free(direct);
    const module = try std.fs.path.join(a, &.{ alias, "mlx/e1" });
    defer a.free(module);
    const native = try std.fs.path.join(a, &.{ direct, "ds4" });
    defer a.free(native);
    const other = try std.fs.path.join(a, &.{ home, "other" });
    defer a.free(other);
    const other_native = try std.fs.path.join(a, &.{ other, "ds4" });
    defer a.free(other_native);
    const pending = (try reserve(a, io, alias, module, 100, 80)).?;
    defer pending.release(io);
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, direct, native, 100, 0));
    try testing.expectEqual(@as(?u64, 100), try availableBudget(a, io, other, other_native, 100, 0));
}
