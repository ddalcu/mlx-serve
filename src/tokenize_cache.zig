//! Per-LoadedModel LRU cache for `chat_mod.formatChat` /
//! `encodeChatViaDs4` / `encodeChatViaLlama` results.
//!
//! Phase 4 #3 win: the long-prompt warm-reuse case spends ~240 ms re-
//! rendering and re-tokenizing a 1813-token prompt on every turn, only to
//! discover the KV cache already has it. The Metal prefill on a full hit
//! is ~30 ms, so tokenize is 7× the actual work. Memoizing the tokenize
//! result over a small LRU drops the second hit to a clone of the cached
//! buffer — sub-millisecond regardless of prompt length.
//!
//! Key derivation is a Wyhash digest over the rendered chat-template
//! inputs:
//!     - per message: role, content
//!     - per tool_call: name + arguments
//!     - tools_json (raw bytes; null contributes a sentinel)
//!     - tool_choice_instruction
//!     - enable_thinking flag
//! Image-bearing messages are intentionally excluded — the vision pipeline
//! re-injects per-request token positions and any cache hit there would
//! point at stale token IDs.
//!
//! Thread safety: a single mutex guards entries. HTTP handlers each run
//! on their own thread; lookups + inserts are short (memcpy + map ops),
//! so a coarse-grained lock is fine.

const std = @import("std");
const chat_mod = @import("chat.zig");

pub const TokenizeCache = struct {
    allocator: std.mem.Allocator,
    /// Maximum entries kept resident. Past this, the least-recently-used
    /// entry is evicted on the next insert. 0 disables the cache entirely.
    capacity: u32,
    /// `std.Io.Mutex` — the codebase uses this everywhere for cross-thread
    /// locking (HTTP request threads consult the cache while the inference
    /// thread services them). `.init` is the default unlocked state.
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayListUnmanaged(Entry) = .empty,
    /// Monotonic counter; each lookup/insert bumps the matching entry's
    /// `last_used` so LRU eviction is well-defined under any ordering.
    counter: u64 = 0,

    pub const Entry = struct {
        key: u64,
        tokens: []u32,
        last_used: u64,
    };

    pub fn init(allocator: std.mem.Allocator, capacity: u32) TokenizeCache {
        return .{ .allocator = allocator, .capacity = capacity };
    }

    pub fn deinit(self: *TokenizeCache) void {
        for (self.entries.items) |*e| self.allocator.free(e.tokens);
        self.entries.deinit(self.allocator);
    }

    /// Compute the cache key for a chat-template render. Returns null if
    /// any input forbids caching (currently: images present on any
    /// message).
    pub fn keyFor(
        messages: []const chat_mod.Message,
        tools_json: ?[]const u8,
        tool_choice_instruction: ?[]const u8,
        enable_thinking: bool,
    ) ?u64 {
        for (messages) |m| if (m.images != null) return null;
        var h = std.hash.Wyhash.init(0xC0DEC0DE);
        for (messages) |m| {
            h.update(m.role);
            h.update("\x1f"); // unit separator — keeps role/content boundary unambiguous
            h.update(m.content);
            h.update("\x1e"); // record separator
            if (m.tool_call_id) |id| h.update(id);
            h.update("\x1e");
            if (m.tool_calls) |tcs| {
                for (tcs) |tc| {
                    h.update(tc.name);
                    h.update("\x1f");
                    h.update(tc.arguments);
                    h.update("\x1f");
                }
            }
            h.update("\x1d"); // group separator between messages
        }
        if (tools_json) |t| h.update(t) else h.update("(no-tools)");
        h.update("\x1e");
        if (tool_choice_instruction) |t| h.update(t) else h.update("(no-tc)");
        h.update("\x1e");
        h.update(if (enable_thinking) "thinking=on" else "thinking=off");
        return h.final();
    }

    /// Return a freshly-allocated clone of the cached token IDs for `key`,
    /// or null on miss. Caller frees the returned slice.
    pub fn get(self: *TokenizeCache, io: std.Io, key: u64, allocator: std.mem.Allocator) !?[]u32 {
        if (self.capacity == 0) return null;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for (self.entries.items) |*e| {
            if (e.key == key) {
                self.counter += 1;
                e.last_used = self.counter;
                return try allocator.dupe(u32, e.tokens);
            }
        }
        return null;
    }

    /// Insert `tokens` under `key`, evicting the LRU entry if at capacity.
    /// The cache makes its own copy of `tokens` — caller retains ownership
    /// of the input slice.
    pub fn put(self: *TokenizeCache, io: std.Io, key: u64, tokens: []const u32) !void {
        if (self.capacity == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // Skip if already present (rare race: two requests with same key
        // both miss + both put). Either copy is fine; keep the first.
        for (self.entries.items) |*e| {
            if (e.key == key) {
                self.counter += 1;
                e.last_used = self.counter;
                return;
            }
        }
        const owned = try self.allocator.dupe(u32, tokens);
        errdefer self.allocator.free(owned);
        if (self.entries.items.len >= self.capacity) {
            // Evict LRU. We always have ≥ 1 entry here because capacity ≥ 1.
            var lru_idx: usize = 0;
            var lru_used: u64 = self.entries.items[0].last_used;
            for (self.entries.items, 0..) |*e, i| {
                if (e.last_used < lru_used) {
                    lru_used = e.last_used;
                    lru_idx = i;
                }
            }
            self.allocator.free(self.entries.items[lru_idx].tokens);
            self.entries.items[lru_idx] = .{ .key = key, .tokens = owned, .last_used = blk: {
                self.counter += 1;
                break :blk self.counter;
            } };
        } else {
            self.counter += 1;
            try self.entries.append(self.allocator, .{ .key = key, .tokens = owned, .last_used = self.counter });
        }
    }
};

test "TokenizeCache key stability" {
    const allocator = std.testing.allocator;
    const m1 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world" },
    };
    const m2 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world" },
    };
    const m3 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world!" }, // different content
    };
    const k1 = TokenizeCache.keyFor(&m1, null, null, false).?;
    const k2 = TokenizeCache.keyFor(&m2, null, null, false).?;
    const k3 = TokenizeCache.keyFor(&m3, null, null, false).?;
    try std.testing.expectEqual(k1, k2);
    try std.testing.expect(k1 != k3);
    _ = allocator;
}

test "TokenizeCache images null key" {
    const m_img = [_]chat_mod.Message{
        .{ .role = "user", .content = "what's in this?", .images = &[_]chat_mod.ImageData{
            .{ .pixels = "", .width = 8, .height = 8 },
        } },
    };
    try std.testing.expect(TokenizeCache.keyFor(&m_img, null, null, false) == null);
}

test "TokenizeCache get/put + LRU eviction" {
    // Use the test-driver's io. std.Io.Threaded(.init) is the standard
    // single-threaded test rig that backs the std.Io.Mutex futex.
    var io_impl: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer io_impl.deinit();
    const io = io_impl.io();

    var c = TokenizeCache.init(std.testing.allocator, 2);
    defer c.deinit();
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 4, 5, 6 };
    const cc = [_]u32{ 7, 8, 9 };
    try c.put(io, 0xA, &a);
    try c.put(io, 0xB, &b);
    // touch A so B is LRU
    if (try c.get(io, 0xA, std.testing.allocator)) |got| std.testing.allocator.free(got);
    try c.put(io, 0xC, &cc);
    // B evicted, A and C present
    try std.testing.expectEqual(@as(usize, 2), c.entries.items.len);
    const got_a = try c.get(io, 0xA, std.testing.allocator) orelse return error.TestExpectedHit;
    defer std.testing.allocator.free(got_a);
    try std.testing.expectEqualSlices(u32, &a, got_a);
    try std.testing.expect((try c.get(io, 0xB, std.testing.allocator)) == null);
}
