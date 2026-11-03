//! Prefix cache for the ANE LM engine: checkpoints of the model state along token sequences.
//!
//! The KV cache can be cut back to any position, the DeltaNet states cannot, so a sequence is a
//! chain of nodes. Each node holds its tokens since its parent, those tokens' KV rows, and the full
//! state at its end until eviction drops it. A restore copies the KV rows of every node from the
//! root down, then the node's state. Both payloads are the engine's bytes; this file never reads them.

const std = @import("std");

pub const Node = struct {
    parent: ?*Node,
    children: std.ArrayList(*Node) = .empty,
    /// Position of the first token (the parent's end).
    start: u32,
    tokens: []u32,
    kv: []u8,
    /// The state at `end()`; null once evicted (the node then only carries KV rows for its children).
    state: ?[]u8,
    last_use: u64 = 0,
    /// Engines whose live state continues this node; a pinned node is never removed.
    pins: u32 = 0,

    pub fn end(self: *const Node) u32 {
        return self.start + @as(u32, @intCast(self.tokens.len));
    }
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    mu: std.Io.Mutex = .init,
    roots: std.ArrayList(*Node) = .empty,
    /// Bytes held by every node's tokens, KV rows and state.
    bytes: u64 = 0,
    states: u32 = 0,
    max_bytes: u64,
    max_states: u32,
    clock: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, max_bytes: u64, max_states: u32) Cache {
        return .{ .allocator = allocator, .max_bytes = max_bytes, .max_states = max_states };
    }

    pub fn deinit(self: *Cache) void {
        for (self.roots.items) |n| self.freeTree(n);
        self.roots.deinit(self.allocator);
    }

    /// The deepest node with a state whose tokens are a prefix of `tokens` and end before its last token
    /// (the request still feeds one token to get its logits).
    pub fn lookup(self: *Cache, tokens: []const u32) ?*Node {
        var best: ?*Node = null;
        deepest(self.roots.items, tokens, &best);
        return best;
    }

    fn deepest(nodes: []const *Node, tokens: []const u32, best: *?*Node) void {
        for (nodes) |n| {
            if (n.end() >= tokens.len or !std.mem.eql(u32, n.tokens, tokens[n.start..n.end()])) continue;
            if (n.state != null and (best.* == null or n.end() > best.*.?.end())) best.* = n;
            deepest(n.children.items, tokens, best);
        }
    }

    /// The child of `parent` (a root when null) holding exactly `tokens`.
    pub fn child(self: *Cache, parent: ?*Node, tokens: []const u32) ?*Node {
        for (self.siblings(parent).items) |n| if (std.mem.eql(u32, n.tokens, tokens)) return n;
        return null;
    }

    /// A new node under `parent` with `tokens` and room for its KV rows and state, for the caller to
    /// fill; `trim` makes room afterwards.
    pub fn add(self: *Cache, parent: ?*Node, tokens: []const u32, kv_len: usize, state_len: usize) !*Node {
        const n = try self.allocator.create(Node);
        errdefer self.allocator.destroy(n);
        const toks = try self.allocator.dupe(u32, tokens);
        errdefer self.allocator.free(toks);
        const kv = try self.allocator.alloc(u8, kv_len);
        errdefer self.allocator.free(kv);
        const state = try self.allocator.alloc(u8, state_len);
        errdefer self.allocator.free(state);
        n.* = .{ .parent = parent, .start = if (parent) |p| p.end() else 0, .tokens = toks, .kv = kv, .state = state };
        try self.siblings(parent).append(self.allocator, n);
        self.bytes += nodeBytes(n);
        self.states += 1;
        self.touch(n);
        return n;
    }

    /// Room for a new state on `n`, whose old one was evicted, for the caller to fill.
    pub fn restate(self: *Cache, n: *Node, len: usize) ![]u8 {
        std.debug.assert(n.state == null);
        const s = try self.allocator.alloc(u8, len);
        n.state = s;
        self.bytes += len;
        self.states += 1;
        self.touch(n);
        return s;
    }

    pub fn touch(self: *Cache, n: *Node) void {
        self.clock += 1;
        n.last_use = self.clock;
    }

    /// Drop the least recently used states until both caps hold. A leaf goes with its rows (and so do
    /// ancestors left without a state, a child or a pin); an inner node keeps its rows for its children.
    pub fn trim(self: *Cache) void {
        while (self.bytes > self.max_bytes or self.states > self.max_states) {
            var lru: ?*Node = null;
            oldest(self.roots.items, &lru);
            const n = lru orelse return;
            self.allocator.free(n.state.?);
            self.bytes -= n.state.?.len;
            self.states -= 1;
            n.state = null;
            self.prune(n);
        }
    }

    fn oldest(nodes: []const *Node, lru: *?*Node) void {
        for (nodes) |n| {
            if (n.state != null and (lru.* == null or n.last_use < lru.*.?.last_use)) lru.* = n;
            oldest(n.children.items, lru);
        }
    }

    /// Remove `n` and then its ancestors while each has no state, no child and no pin.
    fn prune(self: *Cache, n: *Node) void {
        var at: ?*Node = n;
        while (at) |x| {
            if (x.state != null or x.children.items.len > 0 or x.pins > 0) return;
            const list = self.siblings(x.parent);
            for (list.items, 0..) |s, i| if (s == x) {
                _ = list.swapRemove(i);
                break;
            };
            at = x.parent;
            self.bytes -= nodeBytes(x);
            self.destroy(x);
        }
    }

    pub fn pin(_: *Cache, n: *Node) void {
        n.pins += 1;
    }

    pub fn unpin(self: *Cache, n: *Node) void {
        n.pins -= 1;
        self.prune(n);
    }

    fn siblings(self: *Cache, parent: ?*Node) *std.ArrayList(*Node) {
        return if (parent) |p| &p.children else &self.roots;
    }

    fn nodeBytes(n: *const Node) u64 {
        return n.tokens.len * @sizeOf(u32) + n.kv.len + if (n.state) |s| s.len else 0;
    }

    fn destroy(self: *Cache, n: *Node) void {
        self.allocator.free(n.tokens);
        self.allocator.free(n.kv);
        if (n.state) |s| self.allocator.free(s);
        n.children.deinit(self.allocator);
        self.allocator.destroy(n);
    }

    fn freeTree(self: *Cache, n: *Node) void {
        for (n.children.items) |c| self.freeTree(c);
        self.destroy(n);
    }
};

// ── tests ──

const testing = std.testing;

test "lookup: the deepest stateful prefix, ending before the prompt's last token" {
    var c = Cache.init(testing.allocator, 1 << 30, 100);
    defer c.deinit();
    const a = try c.add(null, &.{ 1, 2, 3 }, 0, 1);
    const b = try c.add(a, &.{ 4, 5 }, 0, 1);
    _ = try c.add(a, &.{ 4, 9 }, 0, 1);
    try testing.expectEqual(b, c.lookup(&.{ 1, 2, 3, 4, 5, 6 }).?);
    // A node ending at the prompt's end leaves nothing to feed: its parent serves.
    try testing.expectEqual(a, c.lookup(&.{ 1, 2, 3, 4, 5 }).?);
    try testing.expectEqual(a, c.lookup(&.{ 1, 2, 3, 7 }).?);
    try testing.expect(c.lookup(&.{ 1, 2 }) == null);
    try testing.expect(c.lookup(&.{ 9, 2, 3, 4 }) == null);
    try testing.expectEqual(@as(u32, 5), b.end());
}

test "lookup: a node without a state is passed through to its children" {
    var c = Cache.init(testing.allocator, 1 << 30, 1);
    defer c.deinit();
    const a = try c.add(null, &.{ 1, 2 }, 4, 8);
    const b = try c.add(a, &.{3}, 4, 8);
    c.trim(); // two states, cap one: the older (a) loses its state, keeps its rows for b
    try testing.expect(a.state == null and b.state != null);
    try testing.expectEqual(b, c.lookup(&.{ 1, 2, 3, 4 }).?);
    try testing.expect(c.lookup(&.{ 1, 2, 9 }) == null);
    // The same tokens again give `a` a state back instead of a second node.
    _ = try c.restate(a, 8);
    try testing.expectEqual(a, c.lookup(&.{ 1, 2, 9 }).?);
    try testing.expectEqual(@as(u32, 2), c.states);
}

test "child: the same tokens under the same parent are found, not added twice" {
    var c = Cache.init(testing.allocator, 1 << 30, 100);
    defer c.deinit();
    const a = try c.add(null, &.{ 1, 2 }, 0, 1);
    try testing.expectEqual(a, c.child(null, &.{ 1, 2 }).?);
    try testing.expect(c.child(null, &.{1}) == null);
    try testing.expect(c.child(a, &.{ 1, 2 }) == null);
}

test "trim: least recently used first; a leaf goes with its rows and its stateless ancestors" {
    var c = Cache.init(testing.allocator, 1 << 30, 100);
    defer c.deinit();
    const a = try c.add(null, &.{1}, 10, 100);
    const b = try c.add(a, &.{2}, 10, 100);
    const d = try c.add(null, &.{7}, 10, 100);
    c.touch(a);
    const one = 4 + 10 + 100;
    try testing.expectEqual(@as(u64, 3 * one), c.bytes);
    c.max_bytes = 3 * one - 1;
    c.trim(); // b is the oldest now: a leaf, removed with its rows
    try testing.expectEqual(@as(usize, 0), a.children.items.len);
    try testing.expectEqual(@as(u64, 2 * one), c.bytes);
    c.max_bytes = one;
    c.trim(); // then d (a root leaf)
    try testing.expectEqual(@as(usize, 1), c.roots.items.len);
    try testing.expectEqual(a, c.roots.items[0]);
    _ = d;
    _ = b;
}

test "pin: a pinned node keeps its rows when its state goes, and goes once unpinned" {
    var c = Cache.init(testing.allocator, 1 << 30, 0);
    defer c.deinit();
    const a = try c.add(null, &.{ 1, 2 }, 8, 16);
    c.pin(a);
    c.trim();
    try testing.expect(a.state == null);
    try testing.expectEqual(@as(usize, 1), c.roots.items.len);
    try testing.expectEqual(@as(u64, 8 + 8), c.bytes);
    c.unpin(a);
    try testing.expectEqual(@as(usize, 0), c.roots.items.len);
    try testing.expectEqual(@as(u64, 0), c.bytes);
}

test "unpin: pins from two engines keep a stateless chain until both let go" {
    var c = Cache.init(testing.allocator, 1 << 30, 100);
    defer c.deinit();
    const a = try c.add(null, &.{1}, 4, 8);
    const b = try c.add(a, &.{2}, 4, 8);
    const d = try c.add(b, &.{3}, 4, 8);
    c.pin(d);
    c.pin(b);
    c.max_states = 0;
    c.trim(); // every state goes; the pinned chain stays
    try testing.expectEqual(@as(u32, 0), c.states);
    c.unpin(d); // d goes; b is still pinned, a still has b
    try testing.expectEqual(@as(usize, 0), b.children.items.len);
    try testing.expectEqual(@as(usize, 1), c.roots.items.len);
    c.unpin(b); // b goes, and a with it
    try testing.expectEqual(@as(usize, 0), c.roots.items.len);
    try testing.expectEqual(@as(u64, 0), c.bytes);
}
