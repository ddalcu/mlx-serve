//! Process-wide physical facts and write promises. Only cold setup and explicit worker inventories observe storage.
const std = @import("std");
const testing = std.testing;
const Shared = @import("../util/shared.zig").Shared;
const Stat = std.Io.File.Stat;
const observations = @import("disk_budget_inventory.zig");
pub const Observed = observations.Observed;
pub const Inventory = observations.Inventory;
pub const inventory = observations.inventory;
const State = struct {
    mutex: std.Io.Mutex = .init,
    roots: ?*Base = null,
    first: ?*Reservation = null,
};
pub const reservations_shared = Shared(State, "kv_disk_reservations", .{});
pub const Admission = union(enum) { granted: *Reservation, busy, full, unavailable };
pub const Kind = enum { file, directory, pool };
const accounting = @import("disk_budget_facts.zig");
const Base = accounting.Base;
const owner = accounting.owner;
const putFact = accounting.putFact;
const removePrefix = accounting.removePrefix;
const removeBill = accounting.removeBill;
const addBill = accounting.addBill;
const changeNode = accounting.changeNode;
const clearNodes = accounting.clearNodes;
const invalidate = accounting.invalidate;

/// A model namespace; its physical bill survives model unload while another pool is live.
pub const Pool = struct {
    allocator: std.mem.Allocator,
    ledger: *Base,
    root: []u8,
    alias: []u8,
    refs: usize = 1,
    physical_bytes: u64 = 0,
    exclusive: u64 = 0,
    pending_bytes: u64 = 0,
    serial: u64 = 0,
    next: ?*Pool = null,

    pub fn retain(self: *Pool, io: std.Io) *Pool {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        self.refs += 1;
        return self;
    }

    pub fn close(self: *Pool, io: std.Io) void {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        dropPool(s, self);
    }

    pub fn physical(self: *Pool, io: std.Io) u64 {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        return self.physical_bytes;
    }

    pub fn available(self: *Pool, io: std.Io, cap: u64, own_used: u64) !?u64 {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        if (self.ledger.failed) return error.DiskBudgetUnavailable;
        const other = (self.ledger.physical -| self.exclusive) +| (self.ledger.pending -| self.pending_bytes);
        const n = cap -| other -| own_used;
        return if (n == 0) null else n;
    }

    pub fn isReserved(self: *Pool, io: std.Io, path: []const u8) bool {
        const canonical = observations.canonicalTarget(self.allocator, self.ledger.path, self.ledger.alias, path) catch return true;
        defer self.allocator.free(canonical);
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        var next = s.first;
        while (next) |r| : (next = r.next) {
            if (r.pool.ledger == self.ledger and (within(canonical, r.root) or within(r.root, canonical))) return true;
        }
        return false;
    }

    /// File promises cover a full temporary file in addition to any old final.
    pub fn tryReserve(self: *Pool, io: std.Io, path: []const u8, cap: u64, bytes: ?u64, kind: Kind) !Admission {
        const canonical = try targetPath(self, path);
        errdefer self.allocator.free(canonical);
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const b = self.ledger;
        if (b.failed) {
            self.allocator.free(canonical);
            return .unavailable;
        }
        if (b.refreshing) {
            self.allocator.free(canonical);
            return .busy;
        }
        var next = s.first;
        while (next) |r| : (next = r.next) {
            if (r.pool.ledger == b and (within(r.root, canonical) or within(canonical, r.root)) and
                !(kind == .file and r.kind == .file and std.mem.eql(u8, r.root, canonical)))
            {
                self.allocator.free(canonical);
                return .busy;
            }
        }
        const bill = bytes orelse (cap -| (b.physical -| self.exclusive) -| b.pending);
        const growth = if (kind == .pool) bill -| self.exclusive else bill;
        const zero_pool = kind == .pool and bytes != null and bytes.? == 0;
        if ((bill == 0 and !zero_pool) or growth > cap -| b.physical -| b.pending) {
            self.allocator.free(canonical);
            return if (b.pending != 0 and (bytes == null or growth <= cap -| b.physical)) .busy else .full;
        }
        const r = try self.allocator.create(Reservation);
        errdefer self.allocator.destroy(r);
        r.* = .{
            .pool = self,
            .root = canonical,
            .bytes = bill,
            .remaining = growth,
            .kind = kind,
            .next = s.first,
        };
        self.refs += 1;
        s.first = r;
        b.pending += growth;
        self.pending_bytes += growth;
        return .{ .granted = r };
    }

    /// Facts are keyed by canonical path; repeated publication never invents an old-file credit.
    pub fn publish(self: *Pool, io: std.Io, path: []const u8, before: ?Stat, after: ?Stat, file_lease: ?*Reservation) !void {
        _ = before;
        const canonical = (if (after == null and std.fs.path.isAbsolute(path))
            observations.canonicalTarget(self.allocator, self.ledger.path, self.ledger.alias, path)
        else
            targetPath(self, path)) catch |err| {
            self.unavailable(io);
            return err;
        };
        defer self.allocator.free(canonical);
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const b = self.ledger;
        errdefer invalidate(b);
        var covering = file_lease;
        if (covering == null and after != null) {
            var next = s.first;
            while (next) |r| : (next = r.next) {
                if (r.pool == self and r.kind == .directory and within(canonical, r.root)) {
                    covering = r;
                    break;
                }
            }
        }
        if (covering) |r| {
            if (r.pool != self or !within(canonical, r.root) or (r.kind == .file and !std.mem.eql(u8, canonical, r.root))) return error.InvalidDiskLease;
            if (after) |st| {
                if (r.settled) {
                    const known = b.files.get(canonical) orelse return error.InvalidDiskLease;
                    if (known.stat.inode != st.inode or known.stat.size != st.size) return error.InvalidDiskLease;
                }
                const old = b.files.get(canonical);
                const previous = if (old) |f| (if (f.lease == r) f.stat.size else 0) else 0;
                if (r.visible -| previous +| st.size > r.bytes) return error.DiskLeaseExceeded;
            }
        }
        if (after) |stat| {
            try putFact(b, canonical, self, stat, covering);
            if (covering) |r| if (r.kind == .file) {
                r.settled = true;
            };
        } else removePrefix(b, canonical);
        reprice(s, b);
        b.serial +%= 1;
        self.serial +%= 1;
    }

    pub fn unlinkSucceeded(self: *Pool, io: std.Io, prefix: []const u8) !void {
        try self.publish(io, prefix, null, null, null);
    }

    pub fn unavailable(self: *Pool, io: std.Io) void {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        invalidate(self.ledger);
    }

    /// Native storage owns all mutations of this pool while its whole-pool lease is held.
    /// Observe only that namespace on its worker, then replace its bill atomically.
    pub fn reconcile(self: *Pool, io: std.Io) !void {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        const serial = self.serial;
        s.mutex.unlock(io);
        var observed = inventory(self.allocator, io, self.root) catch |err| {
            self.unavailable(io);
            return err;
        };
        defer observed.deinit(self.allocator);
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const b = self.ledger;
        if (serial != self.serial) return error.DiskBudgetBusy;
        errdefer invalidate(b);
        removePrefix(b, self.root);
        for (observed.items) |f| try putFact(b, f.path, self, f.stat, null);
        var next = s.first;
        while (next) |r| : (next = r.next) {
            if (r.pool == self and r.kind == .pool) {
                if (r.bytes != 0 and self.exclusive > r.bytes) return error.DiskLeaseExceeded;
                r.settled = true;
            }
        }
        reprice(s, b);
        b.serial +%= 1;
        self.serial +%= 1;
    }

    /// Cold root-wide mutation fence. The callback publishes successful changes before returning.
    pub fn sweep(self: *Pool, io: std.Io, context: anytype, comptime action: fn (@TypeOf(context)) anyerror!void) !bool {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        const b = self.ledger;
        if (b.refreshing) {
            s.mutex.unlock(io);
            return false;
        }
        var next = s.first;
        while (next) |r| : (next = r.next) if (r.pool.ledger == b) {
            s.mutex.unlock(io);
            return false;
        };
        b.refreshing = true;
        s.mutex.unlock(io);
        defer {
            s.mutex.lockUncancelable(io);
            b.refreshing = false;
            s.mutex.unlock(io);
        }
        action(context) catch |err| {
            self.unavailable(io);
            return err;
        };
        return true;
    }

    /// Explicit cold/idle inventory. Active promises return busy, never disk-full.
    /// A concurrent eviction publication invalidates this inventory, not admission.
    pub fn refresh(self: *Pool, io: std.Io) !void {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        const b = self.ledger;
        var active = s.first;
        while (active) |r| : (active = r.next) if (r.pool.ledger == b) {
            s.mutex.unlock(io);
            return error.DiskBudgetBusy;
        };
        if (b.refreshing) {
            s.mutex.unlock(io);
            return error.DiskBudgetBusy;
        }
        b.refreshing = true;
        const serial = b.serial;
        s.mutex.unlock(io);
        var observed = inventory(self.allocator, io, b.path) catch |err| {
            s.mutex.lockUncancelable(io);
            b.refreshing = false;
            invalidate(b);
            s.mutex.unlock(io);
            return err;
        };
        defer observed.deinit(self.allocator);
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        b.refreshing = false;
        if (serial != b.serial) return error.DiskBudgetBusy;
        clearNodes(b);
        errdefer invalidate(b);
        for (observed.items) |f| try putFact(b, f.path, owner(b, f.path), f.stat, null);
        b.failed = false;
        b.serial +%= 1;
        var pools = b.pools;
        while (pools) |p| : (pools = p.next) p.serial +%= 1;
    }
};

pub const Reservation = struct {
    pool: *Pool,
    root: []u8,
    bytes: u64,
    remaining: u64,
    visible: u64 = 0,
    settled: bool = false,
    kind: Kind,
    next: ?*Reservation = null,

    pub fn resize(self: *Reservation, io: std.Io, bytes: u64, cap: u64) bool {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const b = self.pool.ledger;
        if (b.failed or b.refreshing or self.settled) return false;
        const needed = switch (self.kind) {
            .file => bytes,
            .directory => bytes -| self.visible,
            .pool => bytes -| self.pool.exclusive,
        };
        if (needed > self.remaining and needed - self.remaining > cap -| b.physical -| b.pending) return false;
        self.bytes = bytes;
        reprice(s, b);
        return true;
    }

    pub fn shrink(self: *Reservation, io: std.Io, bytes: u64) void {
        std.debug.assert(bytes <= self.bytes);
        _ = self.resize(io, bytes, std.math.maxInt(u64));
    }

    pub fn release(self: *Reservation, io: std.Io) void {
        const s = reservations_shared.get();
        s.mutex.lockUncancelable(io);
        defer s.mutex.unlock(io);
        const p = self.pool;
        if (self.kind == .pool and !self.settled) invalidate(p.ledger);
        var link = &s.first;
        while (link.* != self) link = &link.*.?.next;
        link.* = self.next;
        var facts = p.ledger.files.valueIterator();
        while (facts.next()) |f| if (f.lease == self) {
            f.lease = null;
        };
        p.ledger.pending -|= self.remaining;
        p.pending_bytes -|= self.remaining;
        p.allocator.free(self.root);
        p.allocator.destroy(self);
        dropPool(s, p);
    }
};

const within = observations.within;
fn targetPath(p: *Pool, path: []const u8) ![]u8 {
    return observations.canonicalTarget(p.allocator, p.root, p.alias, path);
}
fn reprice(s: *State, b: *Base) void {
    var next = s.first;
    while (next) |r| : (next = r.next) {
        if (r.pool.ledger != b) continue;
        const fresh: u64 = switch (r.kind) {
            .file => if (r.settled) 0 else r.bytes,
            .directory => r.bytes -| r.visible,
            .pool => if (r.settled) 0 else r.bytes -| r.pool.exclusive,
        };
        b.pending = b.pending -| r.remaining;
        r.pool.pending_bytes = r.pool.pending_bytes -| r.remaining;
        b.pending +|= fresh;
        r.pool.pending_bytes +|= fresh;
        r.remaining = fresh;
    }
}
fn dropPool(s: *State, p: *Pool) void {
    p.refs -= 1;
    if (p.refs != 0) return;
    const b = p.ledger;
    var it = b.nodes.valueIterator();
    while (it.next()) |n| {
        if (n.members.get(p)) |refs| {
            removeBill(b, n);
            _ = n.members.remove(p);
            const member = n.members.getOrPut(null) catch {
                invalidate(b);
                continue;
            };
            if (!member.found_existing) member.value_ptr.* = 0;
            member.value_ptr.* += refs;
            addBill(b, n);
        }
    }
    var facts = b.files.valueIterator();
    while (facts.next()) |f| if (f.pool == p) {
        f.pool = null;
    };
    var link = &b.pools;
    while (link.* != p) link = &link.*.?.next;
    link.* = p.next;
    p.allocator.free(p.root);
    p.allocator.free(p.alias);
    p.allocator.destroy(p);
    if (b.pools != null) return;
    var base_link = &s.roots;
    while (base_link.* != b) base_link = &base_link.*.?.next;
    base_link.* = b.next;
    clearNodes(b);
    b.nodes.deinit();
    b.files.deinit();
    b.allocator.free(b.path);
    b.allocator.free(b.alias);
    b.allocator.destroy(b);
}

/// Cold setup only. A shared base is inventoried once; later pools adopt recorded facts.
pub fn openPool(a: std.mem.Allocator, io: std.Io, base: []const u8, root: []const u8) !*Pool {
    const alias = try std.fs.path.resolve(a, &.{base});
    defer a.free(alias);
    const root_alias = try std.fs.path.resolve(a, &.{root});
    defer a.free(root_alias);
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.Io.Dir.cwd().realPathFile(io, alias, &buffer) catch |err| switch (err) {
        error.FileNotFound => 0,
        else => return err,
    };
    const canonical = if (n == 0) alias else buffer[0..n];
    const canonical_root = try observations.canonicalTarget(a, canonical, alias, root_alias);
    defer a.free(canonical_root);
    const s = reservations_shared.get();
    s.mutex.lockUncancelable(io);
    var locked = true;
    defer if (locked) s.mutex.unlock(io);
    var b = s.roots;
    while (b) |v| : (b = v.next) if (std.mem.eql(u8, v.path, canonical)) break;
    const fresh = b == null;
    if (fresh) {
        const v = try a.create(Base);
        errdefer a.destroy(v);
        const path = try a.dupe(u8, canonical);
        errdefer a.free(path);
        v.* = .{ .allocator = a, .path = path, .alias = try a.dupe(u8, alias), .nodes = .init(a), .files = .init(a), .failed = true, .next = s.roots };
        s.roots = v;
        b = v;
    }
    const ledger = b.?;
    var transferred = false;
    errdefer if (!transferred and fresh and ledger.pools == null) {
        s.roots = ledger.next;
        ledger.nodes.deinit();
        ledger.files.deinit();
        a.free(ledger.path);
        a.free(ledger.alias);
        a.destroy(ledger);
    };
    if (ledger.refreshing) return error.DiskBudgetBusy;
    var existing = ledger.pools;
    while (existing) |p| : (existing = p.next) {
        if (std.mem.eql(u8, p.root, canonical_root)) {
            p.refs += 1;
            return p;
        }
        if (within(canonical_root, p.root) or within(p.root, canonical_root)) return error.OverlappingDiskPools;
    }
    const p = try a.create(Pool);
    errdefer if (!transferred) a.destroy(p);
    const owned = try a.dupe(u8, canonical_root);
    errdefer if (!transferred) a.free(owned);
    p.* = .{ .allocator = a, .ledger = ledger, .root = owned, .alias = try a.dupe(u8, root_alias), .next = ledger.pools };
    ledger.pools = p;
    if (!fresh) {
        var it = ledger.files.iterator();
        while (it.next()) |entry| if (within(entry.key_ptr.*, p.root)) {
            const f = entry.value_ptr;
            changeNode(ledger, f.pool, f.stat, false) catch unreachable;
            changeNode(ledger, p, f.stat, true) catch |err| {
                invalidate(ledger);
                transferred = true;
                dropPool(s, p);
                return err;
            };
            f.pool = p;
        };
        return p;
    }
    transferred = true;
    s.mutex.unlock(io);
    locked = false;
    // All owned allocations now belong to the registered pool, including on refresh failure.
    p.refresh(io) catch |err| {
        p.close(io);
        return err;
    };
    return p;
}

fn findPool(s: *State, base: []const u8, path: []const u8) ?*Pool {
    var b = s.roots;
    while (b) |v| : (b = v.next) {
        if (!std.mem.eql(u8, base, v.path) and !std.mem.eql(u8, base, v.alias)) continue;
        var p = v.pools;
        while (p) |pool| : (p = pool.next) {
            if (within(path, pool.root) or within(path, pool.alias)) return pool;
        }
    }
    return null;
}
/// Returns a retained pool; the caller closes its reference.
pub fn registeredPool(io: std.Io, base: []const u8, root: []const u8) !*Pool {
    const s = reservations_shared.get();
    s.mutex.lockUncancelable(io);
    defer s.mutex.unlock(io);
    const p = findPool(s, base, root) orelse return error.DiskBudgetUninitialized;
    p.refs += 1;
    return p;
}
pub fn availableBudget(_: std.mem.Allocator, io: std.Io, base: []const u8, own_root: []const u8, cap: u64, own_used: u64) !?u64 {
    const p = try registeredPool(io, base, own_root);
    defer p.close(io);
    return p.available(io, cap, own_used);
}
pub fn reserve(_: std.mem.Allocator, io: std.Io, base: []const u8, root: []const u8, cap: u64, bytes: ?u64) !?*Reservation {
    const p = try registeredPool(io, base, root);
    defer p.close(io);
    return switch (try p.tryReserve(io, root, cap, bytes, if (bytes == null) .pool else .directory)) {
        .granted => |r| r,
        .full => null,
        .busy => error.DiskBudgetBusy,
        .unavailable => error.DiskBudgetUnavailable,
    };
}
pub const physicalBytes = observations.physicalBytes;

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
    const mlx_pool = try openPool(a, io, base, mlx_root);
    defer mlx_pool.close(io);
    const ds4_pool = try openPool(a, io, base, ds4_root);
    defer ds4_pool.close(io);
    try testing.expectEqual(@as(?u64, 50), try availableBudget(a, io, base, mlx_root, 100, 0));
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, mlx_root, 100, 30));
    try testing.expectEqual(@as(?u64, 60), try availableBudget(a, io, base, ds4_root, 100, 0));
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, base, ds4_root, 100, 40));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 50, 0));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 100, 50));
    try testing.expectEqual(@as(?u64, null), try availableBudget(a, io, base, mlx_root, 0, 0));
    // A native save being replaced is measured by its current size, without interpreting metadata.
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/model/kv/save.tmp", .data = &@as([70]u8, @splat(4)) });
    try ds4_pool.publish(io, "model/kv/save.tmp", null, try tmp.dir.statFile(io, "ds4/model/kv/save.tmp", .{}), null);
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
    const pool = try openPool(a, io, base, own);
    defer pool.close(io);
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
    const pool = try openPool(a, io, base, own);
    defer pool.close(io);
    // The model's namespace need not exist before its startup allowance is measured.
    try testing.expectEqual(@as(?u64, 94), try availableBudget(a, io, base, own, 100, 0));
    try tmp.dir.createDirPath(io, "ssd/kv-cache/0000000000000001");
    try tmp.dir.writeFile(io, .{ .sub_path = "ssd/kv-cache/0000000000000001/chunk", .data = "owned" });
    try pool.publish(io, "chunk", null, try tmp.dir.statFile(io, "ssd/kv-cache/0000000000000001/chunk", .{}), null);
    try testing.expectEqual(@as(?u64, 94), try availableBudget(a, io, base, own, 100, 0));
    try testing.expectEqual(@as(u64, 11), try physicalBytes(a, io, base, null));
}

test "shared disk reservations charge published remnants and retain unknown files until refresh" {
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
    const mlx_pool = try openPool(a, io, base, module);
    defer mlx_pool.close(io);
    const native_pool = try openPool(a, io, base, native);
    defer native_pool.close(io);
    {
        const reservation = (try reserve(a, io, base, module, 100, 80)).?;
        defer reservation.release(io);
        try testing.expectEqual(@as(?u64, 20), try native_pool.available(io, 100, 0));
        try tmp.dir.writeFile(io, .{ .sub_path = "mlx/e1/part.tmp", .data = &@as([30]u8, @splat(1)) });
        try mlx_pool.publish(io, "part.tmp", null, try tmp.dir.statFile(io, "mlx/e1/part.tmp", .{}), null);
        try testing.expectEqual(@as(?u64, 20), try native_pool.available(io, 100, 0));
        try tmp.dir.writeFile(io, .{ .sub_path = "mlx/e1/part.tmp", .data = &@as([10]u8, @splat(2)) });
        try mlx_pool.publish(io, "part.tmp", null, try tmp.dir.statFile(io, "mlx/e1/part.tmp", .{}), null);
        try testing.expectEqual(@as(u64, 80), native_pool.ledger.physical + native_pool.ledger.pending);
        try tmp.dir.writeFile(io, .{ .sub_path = "unindexed", .data = "unknown" });
        try testing.expectEqual(@as(?u64, 20), try native_pool.available(io, 100, 0));
        try testing.expectError(error.DiskBudgetBusy, mlx_pool.refresh(io));
    }
    try mlx_pool.refresh(io);
    try testing.expectEqual(@as(?u64, 83), try native_pool.available(io, 100, 0));
    try tmp.dir.deleteFile(io, "mlx/e1/part.tmp");
    try mlx_pool.unlinkSucceeded(io, "part.tmp");
    try testing.expectEqual(@as(?u64, 93), try native_pool.available(io, 100, 0));
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
    const p1 = try openPool(a, io, base, roots[0]);
    defer p1.close(io);
    const p2 = try openPool(a, io, base, roots[1]);
    defer p2.close(io);
    const Worker = struct {
        fn run(base_path: []const u8, root: []const u8, out: *?*Reservation, err: *?anyerror) void {
            out.* = reserve(testing.allocator, testing.io, base_path, root, 100, 80) catch |e| {
                if (e != error.DiskBudgetBusy) err.* = e;
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
    const p1 = try openPool(a, io, alias, module);
    defer p1.close(io);
    const p2 = try openPool(a, io, direct, native);
    defer p2.close(io);
    const p3 = try openPool(a, io, other, other_native);
    defer p3.close(io);
    const pending = (try reserve(a, io, alias, module, 100, 80)).?;
    defer pending.release(io);
    try testing.expectEqual(@as(?u64, 20), try availableBudget(a, io, direct, native, 100, 0));
    try testing.expectEqual(@as(?u64, 100), try availableBudget(a, io, other, other_native, 100, 0));
}

test "shared disk ledger keeps hard-link facts across pool close and final base teardown" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx");
    try tmp.dir.createDirPath(io, "ds4");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/a", .data = "1234567890" });
    try std.Io.Dir.hardLink(tmp.dir, "mlx/a", tmp.dir, "ds4/b", io, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "unknown", .data = "12345" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const mx = try std.fs.path.join(a, &.{ base, "mlx" });
    defer a.free(mx);
    const ds = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(ds);
    const mlx_pool = try openPool(a, io, base, mx);
    const native = try openPool(a, io, base, ds);
    const stat = try tmp.dir.statFile(io, "mlx/a", .{});
    try mlx_pool.publish(io, "a", stat, stat, null);
    try mlx_pool.publish(io, "a", stat, stat, null);
    try testing.expectEqual(@as(u64, 15), native.ledger.physical);
    mlx_pool.close(io);
    try testing.expectEqual(@as(u64, 15), native.ledger.physical);
    const reopened = try openPool(a, io, base, mx);
    try tmp.dir.deleteFile(io, "mlx/a");
    try reopened.publish(io, "a", stat, null, null);
    try reopened.publish(io, "a", stat, null, null);
    try testing.expectEqual(@as(u64, 15), native.ledger.physical);
    reopened.close(io);
    try tmp.dir.deleteFile(io, "ds4/b");
    try native.publish(io, "b", stat, null, null);
    try testing.expectEqual(@as(u64, 5), native.ledger.physical);
    try testing.expectEqual(@as(u32, 1), native.ledger.nodes.count());
    native.close(io);
    try testing.expect(reservations_shared.get().roots == null);
    try testing.expect(reservations_shared.get().first == null);
}

test "shared disk file leases charge full duplicate temporaries and survive owner close" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx");
    try tmp.dir.createDirPath(io, "ds4");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/final", .data = &@as([30]u8, @splat(1)) });
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/keep", .data = &@as([20]u8, @splat(2)) });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const mx = try std.fs.path.join(a, &.{ base, "mlx" });
    defer a.free(mx);
    const ds = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(ds);
    const mlx_pool = try openPool(a, io, base, mx);
    const native = try openPool(a, io, base, ds);
    defer native.close(io);
    const first = (try mlx_pool.tryReserve(io, "final", 100, 30, .file)).granted;
    const second = (try mlx_pool.tryReserve(io, "final", 100, 20, .file)).granted;
    try testing.expectEqual(@as(u64, 50), native.ledger.pending);
    try testing.expectEqual(Admission.busy, try native.tryReserve(io, "new", 100, 1, .file));
    try testing.expect(native.isReserved(io, first.root));
    mlx_pool.close(io);
    const old = try tmp.dir.statFile(io, "mlx/final", .{});
    try first.pool.publish(io, "final", old, old, first);
    first.release(io);
    try testing.expectEqual(@as(u64, 50), native.ledger.physical);
    try testing.expectEqual(@as(u64, 20), native.ledger.pending);
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/new.tmp", .data = &@as([20]u8, @splat(3)) });
    try tmp.dir.rename("mlx/new.tmp", tmp.dir, "mlx/final", io);
    try second.pool.publish(io, "final", old, try tmp.dir.statFile(io, "mlx/final", .{}), second);
    try testing.expectEqual(@as(u64, 40), native.ledger.physical);
    try testing.expectEqual(@as(u64, 0), native.ledger.pending);
    second.release(io);
    try testing.expectEqual(@as(u64, 40), native.ledger.physical);
    try testing.expectEqual(@as(?u64, 60), try native.available(io, 100, 20));
    for (0..32) |_| {
        try tmp.dir.writeFile(io, .{ .sub_path = "ds4/churn", .data = "x" });
        const st = try tmp.dir.statFile(io, "ds4/churn", .{});
        try native.publish(io, "churn", null, st, null);
        try tmp.dir.deleteFile(io, "ds4/churn");
        try native.publish(io, "churn", st, null, null);
    }
    try testing.expectEqual(@as(u32, 2), native.ledger.nodes.count());
}

test "shared disk native allowance holds until inventory publication and fails closed on abandonment" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx");
    try tmp.dir.createDirPath(io, "ds4");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/keep", .data = &@as([30]u8, @splat(1)) });
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/state", .data = &@as([20]u8, @splat(2)) });
    try tmp.dir.writeFile(io, .{ .sub_path = "unknown", .data = &@as([10]u8, @splat(3)) });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const mx = try std.fs.path.join(a, &.{ base, "mlx" });
    defer a.free(mx);
    const ds = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(ds);
    const mlx_pool = try openPool(a, io, base, mx);
    defer mlx_pool.close(io);
    const native = try openPool(a, io, base, ds);
    defer native.close(io);
    const writer = (try mlx_pool.tryReserve(io, "new", 100, 10, .file)).granted;
    const allowance = (try native.tryReserve(io, ".", 100, null, .pool)).granted;
    try testing.expectEqual(@as(u64, 50), allowance.bytes);
    try testing.expectEqual(@as(u64, 100), native.ledger.physical + native.ledger.pending);
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx/new", .data = &@as([10]u8, @splat(4)) });
    try mlx_pool.publish(io, "new", null, try tmp.dir.statFile(io, "mlx/new", .{}), writer);
    writer.release(io);
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/state", .data = &@as([40]u8, @splat(5)) });
    try testing.expectEqual(@as(u64, 100), native.ledger.physical + native.ledger.pending);
    try native.reconcile(io);
    try testing.expect(allowance.settled);
    try testing.expectEqual(@as(u64, 90), native.ledger.physical);
    try testing.expectEqual(@as(u64, 0), native.ledger.pending);
    allowance.release(io);
    try testing.expectEqual(@as(?u64, 10), try mlx_pool.available(io, 100, 40));
    {
        const read_only = (try native.tryReserve(io, ".", 0, 0, .pool)).granted;
        defer read_only.release(io);
        try testing.expectEqual(Admission.busy, try native.tryReserve(io, ".", 100, null, .pool));
        try native.reconcile(io);
        try testing.expectEqual(@as(u64, 90), native.ledger.physical);
        try testing.expectEqual(@as(u64, 0), native.ledger.pending);
    }
    const abandoned = (try native.tryReserve(io, ".", 100, null, .pool)).granted;
    abandoned.release(io);
    try testing.expectError(error.DiskBudgetUnavailable, mlx_pool.available(io, 100, 0));
    try native.refresh(io);
    try testing.expectEqual(@as(u64, 90), native.ledger.physical);
}

test "shared disk hot APIs operate with every filesystem operation disabled" {
    const a = testing.allocator;
    const io = testing.io;
    const hot = std.Io.failing;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "own");
    try tmp.dir.writeFile(io, .{ .sub_path = "own/file", .data = "x" });
    const stat = try tmp.dir.statFile(io, "own/file", .{});
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const root = try std.fs.path.join(a, &.{ base, "own" });
    defer a.free(root);
    const pool = try openPool(a, io, base, root);
    _ = pool.retain(hot);
    pool.close(hot);
    const lease = (try pool.tryReserve(hot, "file", 100, 1, .file)).granted;
    try testing.expect(lease.resize(hot, 2, 100));
    lease.shrink(hot, 1);
    try testing.expect(pool.isReserved(hot, lease.root));
    try testing.expectEqual(@as(u64, 1), pool.physical(hot));
    try testing.expectEqual(@as(?u64, 99), try pool.available(hot, 100, 1));
    try pool.publish(hot, "file", stat, stat, lease);
    try pool.publish(hot, "file", stat, stat, lease);
    try pool.publish(hot, "missing", stat, null, null);
    lease.release(hot);
    try testing.expectEqual(@as(u32, 1), pool.ledger.nodes.count());
    pool.close(hot);
    try testing.expect(reservations_shared.get().roots == null);
}

test "shared disk sweep fences sibling deletion against new admissions" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx");
    try tmp.dir.createDirPath(io, "ds4");
    try tmp.dir.writeFile(io, .{ .sub_path = "ds4/victim", .data = "native" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const mx = try std.fs.path.join(a, &.{ base, "mlx" });
    defer a.free(mx);
    const ds = try std.fs.path.join(a, &.{ base, "ds4" });
    defer a.free(ds);
    const pool = try openPool(a, io, base, mx);
    defer pool.close(io);
    const native = try openPool(a, io, base, ds);
    defer native.close(io);
    const Context = struct {
        pool: *Pool,
        native: *Pool,
        dir: std.Io.Dir,
        fn run(ctx: *@This()) !void {
            try testing.expectEqual(Admission.busy, try ctx.native.tryReserve(testing.io, "new", 100, 1, .file));
            var files = try inventory(testing.allocator, testing.io, ctx.native.root);
            defer files.deinit(testing.allocator);
            try ctx.dir.deleteFile(testing.io, "ds4/victim");
            for (files.items) |f| try ctx.pool.publish(testing.io, f.path, f.stat, null, null);
        }
    };
    var context: Context = .{ .pool = pool, .native = native, .dir = tmp.dir };
    const lease = (try native.tryReserve(io, "new", 100, 1, .file)).granted;
    try testing.expect(!try pool.sweep(io, &context, Context.run));
    lease.release(io);
    try testing.expect(try pool.sweep(io, &context, Context.run));
    try testing.expectEqual(@as(u64, 0), pool.ledger.physical);
    const next = (try native.tryReserve(io, "new", 100, 100, .file)).granted;
    next.release(io);
}

test "shared disk paused inventory cannot erase a concurrent accounting failure" {
    const Pause = struct {
        threadlocal var current: ?*@This() = null;
        pool: *Pool,
        io: std.Io,
        native: bool,
        mutex: std.Io.Mutex = .init,
        changed: std.Io.Condition = .init,
        paused: bool = false,
        resumed: bool = false,
        finished: bool = false,
        failure: ?anyerror = null,

        fn read(ud: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
            const n = try testing.io.vtable.dirRead(ud, reader, entries);
            if (n == 0) if (current) |self| {
                self.mutex.lockUncancelable(testing.io);
                defer self.mutex.unlock(testing.io);
                if (!self.paused) {
                    self.paused = true;
                    self.changed.broadcast(testing.io);
                    while (!self.resumed) self.changed.waitUncancelable(testing.io, &self.mutex);
                }
            };
            return n;
        }

        fn run(self: *@This()) void {
            current = self;
            defer current = null;
            const result = if (self.native) self.pool.reconcile(self.io) else self.pool.refresh(self.io);
            result catch |err| {
                self.failure = err;
            };
            self.mutex.lockUncancelable(testing.io);
            defer self.mutex.unlock(testing.io);
            self.finished = true;
            self.changed.broadcast(testing.io);
        }

        fn releasePause(self: *@This()) void {
            self.mutex.lockUncancelable(testing.io);
            defer self.mutex.unlock(testing.io);
            self.resumed = true;
            self.changed.broadcast(testing.io);
        }
    };
    const a = testing.allocator;
    const io = testing.io;
    for ([_]bool{ false, true }) |native| {
        for ([_]bool{ false, true }) |publication_failure| {
            var tmp = testing.tmpDir(.{});
            defer tmp.cleanup();
            try tmp.dir.createDirPath(io, "pool");
            try tmp.dir.writeFile(io, .{ .sub_path = "pool/donor", .data = &@as([80]u8, @splat(1)) });
            const stat = try tmp.dir.statFile(io, "pool/donor", .{});
            var buf: [std.fs.max_path_bytes]u8 = undefined;
            const base = buf[0..try tmp.dir.realPath(io, &buf)];
            const root = try std.fs.path.join(a, &.{ base, "pool" });
            defer a.free(root);
            const pool = try openPool(a, io, base, root);
            defer pool.close(io);
            var lease: ?*Reservation = if (native) (try pool.tryReserve(io, ".", 100, null, .pool)).granted else null;
            defer if (lease) |r| r.release(io);
            var vtable = io.vtable.*;
            vtable.dirRead = Pause.read;
            var pause: Pause = .{ .pool = pool, .io = .{ .userdata = io.userdata, .vtable = &vtable }, .native = native };
            const worker = try std.Thread.spawn(.{}, Pause.run, .{&pause});
            var joined = false;
            defer if (!joined) {
                pause.releasePause();
                worker.join();
            };
            pause.mutex.lockUncancelable(io);
            while (!pause.paused and !pause.finished) pause.changed.waitUncancelable(io, &pause.mutex);
            const paused = pause.paused;
            pause.mutex.unlock(io);
            try testing.expect(paused);

            // The inventory has passed this directory. An unrecorded alias must
            // keep admission unavailable until a fresh inventory observes it.
            try std.Io.Dir.hardLink(tmp.dir, "pool/donor", tmp.dir, "pool/alias", io, .{});
            if (publication_failure) {
                var failing = testing.FailingAllocator.init(a, .{ .fail_index = 0 });
                const original = pool.ledger.allocator;
                pool.ledger.allocator = failing.allocator();
                const result = pool.publish(io, "alias", null, stat, null);
                pool.ledger.allocator = original;
                try testing.expectError(error.OutOfMemory, result);
            } else pool.unavailable(io);
            pause.releasePause();
            worker.join();
            joined = true;
            try testing.expectEqual(@as(?anyerror, error.DiskBudgetBusy), pause.failure);
            try testing.expectError(error.DiskBudgetUnavailable, pool.available(io, 100, 0));
            if (lease) |r| {
                try testing.expect(!r.settled);
                r.release(io);
                lease = null;
            }
            try tmp.dir.deleteFile(io, "pool/donor");
            try pool.unlinkSucceeded(io, "donor");
            try testing.expectEqual(Admission.unavailable, try pool.tryReserve(io, "next", 100, 100, .file));
            try pool.refresh(io);
            try testing.expectEqual(@as(u64, 80), pool.physical(io));
            try testing.expectEqual(Admission.full, try pool.tryReserve(io, "next", 100, 21, .file));
        }
    }
}
