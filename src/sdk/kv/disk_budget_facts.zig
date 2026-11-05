//! Recorded path/inode facts and their physical bills. All mutations hold the budget's Shared mutex.
const std = @import("std");
const Pool = @import("disk_budget.zig").Pool;
const Reservation = @import("disk_budget.zig").Reservation;
const Stat = std.Io.File.Stat;
const Inode = std.Io.File.INode;
const within = @import("disk_budget_inventory.zig").within;

const Node = struct {
    bytes: u64,
    members: std.AutoHashMap(?*Pool, usize),
};
const Fact = struct { stat: Stat, pool: ?*Pool, lease: ?*Reservation = null };
pub const Base = struct {
    allocator: std.mem.Allocator,
    path: []u8,
    alias: []u8,
    nodes: std.AutoHashMap(Inode, Node),
    files: std.StringHashMap(Fact),
    pools: ?*Pool = null,
    physical: u64 = 0,
    pending: u64 = 0,
    serial: u64 = 0,
    refreshing: bool = false,
    failed: bool = false,
    next: ?*Base = null,
};

/// Caller holds the shared mutex. Even repeated failures invalidate inventories
/// already running against the base or any native pool.
pub fn invalidate(b: *Base) void {
    b.failed = true;
    b.serial +%= 1;
    var pools = b.pools;
    while (pools) |p| : (pools = p.next) p.serial +%= 1;
}

pub fn owner(b: *Base, path: []const u8) ?*Pool {
    var p = b.pools;
    while (p) |v| : (p = v.next) if (within(path, v.root)) return v;
    return null;
}
pub fn putFact(b: *Base, path: []const u8, p: ?*Pool, stat: Stat, lease: ?*Reservation) !void {
    if (stat.kind != .file) return;
    const owned = try b.allocator.dupe(u8, path);
    errdefer b.allocator.free(owned);
    try b.files.ensureUnusedCapacity(1);
    if (b.nodes.get(stat.inode)) |node| if (node.bytes != stat.size) {
        var facts = b.files.valueIterator();
        while (facts.next()) |f| if (f.stat.inode == stat.inode) {
            if (f.lease) |r| r.visible = r.visible -| f.stat.size +| stat.size;
            f.stat.size = stat.size;
        };
    };
    if (b.files.fetchRemove(path)) |old| {
        if (old.value.lease) |r| r.visible -|= old.value.stat.size;
        try changeNode(b, old.value.pool, old.value.stat, false);
        b.allocator.free(old.key);
    }
    try changeNode(b, p, stat, true);
    b.files.putAssumeCapacity(owned, .{ .stat = stat, .pool = p, .lease = lease });
    if (lease) |r| r.visible +|= stat.size;
}
pub fn removePrefix(b: *Base, prefix: []const u8) void {
    var it = b.files.iterator();
    while (it.next()) |entry| if (within(entry.key_ptr.*, prefix)) {
        const old = b.files.fetchRemove(entry.key_ptr.*).?;
        if (old.value.lease) |r| r.visible -|= old.value.stat.size;
        if (old.value.pool) |p| p.serial +%= 1;
        changeNode(b, old.value.pool, old.value.stat, false) catch {
            invalidate(b);
        };
        b.allocator.free(old.key);
    };
}
pub fn removeBill(b: *Base, n: *Node) void {
    if (n.members.count() == 0) return;
    b.physical -|= n.bytes;
    var it = n.members.keyIterator();
    while (it.next()) |p| if (p.*) |v| {
        v.physical_bytes -|= n.bytes;
        if (n.members.count() == 1) v.exclusive -|= n.bytes;
    };
}
pub fn addBill(b: *Base, n: *Node) void {
    if (n.members.count() == 0) return;
    b.physical +|= n.bytes;
    var it = n.members.keyIterator();
    while (it.next()) |p| if (p.*) |v| {
        v.physical_bytes +|= n.bytes;
        if (n.members.count() == 1) v.exclusive +|= n.bytes;
    };
}
pub fn changeNode(b: *Base, p: ?*Pool, stat: Stat, add: bool) !void {
    if (stat.kind != .file) return;
    const n = if (add) blk: {
        const found = try b.nodes.getOrPut(stat.inode);
        if (!found.found_existing) found.value_ptr.* = .{ .bytes = stat.size, .members = .init(b.allocator) };
        break :blk found.value_ptr;
    } else b.nodes.getPtr(stat.inode) orelse return;
    removeBill(b, n);
    if (add) {
        n.bytes = stat.size;
        const member = try n.members.getOrPut(p);
        if (!member.found_existing) member.value_ptr.* = 0;
        member.value_ptr.* += 1;
    } else if (n.members.getPtr(p)) |refs| {
        if (refs.* > 1) refs.* -= 1 else _ = n.members.remove(p);
    }
    addBill(b, n);
    if (n.members.count() == 0) {
        n.members.deinit();
        _ = b.nodes.remove(stat.inode);
    }
}
pub fn clearNodes(b: *Base) void {
    var paths = b.files.keyIterator();
    while (paths.next()) |path| b.allocator.free(path.*);
    b.files.clearRetainingCapacity();
    var it = b.nodes.valueIterator();
    while (it.next()) |n| n.members.deinit();
    b.nodes.clearRetainingCapacity();
    b.physical = 0;
    var p = b.pools;
    while (p) |v| : (p = v.next) {
        v.physical_bytes = 0;
        v.exclusive = 0;
    }
}
