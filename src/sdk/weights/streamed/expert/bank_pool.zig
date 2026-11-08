//! One buffer both phases' slot rows carve (the stream's `BankPool`), so a phase change re-partitions it
//! instead of freeing one phase's rows and freshly allocating the other's (o2a1's split: 176-191 ms prompt to decode and
//! 94-101 ms back, about 13-15 ms a GiB of fresh MLX allocation).
//!
//! The prompt layout, from byte 0: the transient scratch of each geometry (5 windows of 48 rows: 5,082 MiB on the
//! 2.81 bpw pack), then the source's extra arrays (the two wave sets' twelve regions, scratch_waves.zig: 875 MiB).
//! The decode layout over the same bytes: window 0 of each geometry, then whole ext component arrays, largest first,
//! each while it fits (the rest allocated at the grow as before). Neither phase holds more than without the pool: the
//! prompt layout is what a prompt holds from its first wide call to its end (the windows and the wave sets), and the
//! decode layout is part of decode's own rows. Every pooled array is its own MTLBuffer (the host's `HostGpu.alias`): MLX
//! orders each apart, as separate allocations. Exact: the same plans, rows, kernels and words; a pooled row is read
//! only after its record landed, and a range changes hands only at a phase change, after the stream's drains and the
//! source's synchronize.
//!
//! This file: the plan (MLX-free).

const std = @import("std");

/// One array a layout places: its bytes (rounded up to the page in the layout); `plan` sets its offset and whether the
/// pool holds it (only a decode ext item may stay out: it is allocated at the grow as without the pool).
pub const Item = struct {
    bytes: u64,
    offset: u64 = 0,
    pooled: bool = false,
};

pub const Plan = struct {
    /// The pool's bytes: the prompt layout's.
    bytes: u64 = 0,
    /// The bytes each layout covers from 0.
    prompt_end: u64 = 0,
    decode_end: u64 = 0,
    /// Decode's ext items: held by the pool, and allocated at the grow.
    pooled: u32 = 0,
    pooled_bytes: u64 = 0,
    fresh: u32 = 0,
    fresh_bytes: u64 = 0,
};

pub const PlanError = error{ BankPoolEmpty, BankPoolWindowTooLarge, BankPoolOrder };

/// The two layouts from byte 0, every item on `page`: `prompt` in order, all held (the pool is their bytes); `window`
/// in order, all held, within them; then `ext`, largest first (ties: the earlier item), each held while it fits under
/// the pool's bytes, the others left out. `order` (one entry per ext item) is scratch.
pub fn plan(prompt: []Item, window: []Item, ext: []Item, page: u64, order: []u32) PlanError!Plan {
    if (order.len != ext.len) return error.BankPoolOrder;
    var p: Plan = .{};
    var off: u64 = 0;
    for (prompt) |*it| {
        it.offset = off;
        it.pooled = true;
        off += std.mem.alignForward(u64, it.bytes, page);
    }
    if (off == 0) return error.BankPoolEmpty;
    p.bytes = off;
    p.prompt_end = off;
    var d: u64 = 0;
    for (window) |*it| {
        it.offset = d;
        it.pooled = true;
        d += std.mem.alignForward(u64, it.bytes, page);
    }
    if (d > p.bytes) return error.BankPoolWindowTooLarge;
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, ext, struct {
        fn lt(e: []Item, x: u32, y: u32) bool {
            if (e[x].bytes != e[y].bytes) return e[x].bytes > e[y].bytes;
            return x < y;
        }
    }.lt);
    for (order) |i| {
        const it = &ext[i];
        const r = std.mem.alignForward(u64, it.bytes, page);
        if (it.bytes > 0 and d + r <= p.bytes) {
            it.offset = d;
            it.pooled = true;
            d += r;
            p.pooled += 1;
            p.pooled_bytes += it.bytes;
        } else {
            it.offset = 0;
            it.pooled = false;
            p.fresh += 1;
            p.fresh_bytes += it.bytes;
        }
    }
    p.decode_end = d;
    return p;
}

/// The `[bank-pool] plan` line (MiB for bytes, KiB for the decode layout's slack under the pool).
const testing = std.testing;

fn disjoint(items: []const Item, page: u64) !void {
    for (items, 0..) |a, i| {
        if (!a.pooled) continue;
        try testing.expect(a.offset % page == 0);
        for (items[i + 1 ..]) |b| {
            if (!b.pooled) continue;
            const a_end = a.offset + std.mem.alignForward(u64, a.bytes, page);
            const b_end = b.offset + std.mem.alignForward(u64, b.bytes, page);
            try testing.expect(a_end <= b.offset or b_end <= a.offset);
        }
    }
}

test "bank_pool: each layout's items are page-aligned and disjoint, decode's within the prompt's bytes, largest first" {
    const page: u64 = 16384;
    // The prompt: two transient components and two extra arrays; decode: window 0's two, then ext arrays of 3 layers.
    var prompt = [_]Item{ .{ .bytes = 100 * page }, .{ .bytes = 40 * page + 7 }, .{ .bytes = 30 * page }, .{ .bytes = 5 } };
    var window = [_]Item{ .{ .bytes = 8 * page }, .{ .bytes = 3 * page + 1 } };
    var ext = [_]Item{ .{ .bytes = 50 * page }, .{ .bytes = 20 * page }, .{ .bytes = 50 * page }, .{ .bytes = 60 * page + 3 }, .{ .bytes = 2 * page }, .{ .bytes = 0 } };
    var order: [ext.len]u32 = undefined;
    const p = try plan(&prompt, &window, &ext, page, &order);
    // The pool is the prompt layout: 100 + 41 + 30 + 1 pages.
    try testing.expectEqual(@as(u64, 172 * page), p.bytes);
    try testing.expectEqual(p.bytes, p.prompt_end);
    for (prompt) |it| try testing.expect(it.pooled);
    for (window) |it| try testing.expect(it.pooled);
    try disjoint(&prompt, page);
    var decode: [window.len + ext.len]Item = undefined;
    @memcpy(decode[0..window.len], &window);
    @memcpy(decode[window.len..], &ext);
    try disjoint(&decode, page);
    // Window 0 takes 12 pages; then 61 (the largest), 50, 50 (the earlier first) = 173 > 172: the second 50 stays out;
    // 20 and 2 fit (12 + 61 + 50 + 20 + 2 = 145); the empty item stays out.
    try testing.expect(ext[3].pooled and ext[0].pooled and !ext[2].pooled and ext[1].pooled and ext[4].pooled and !ext[5].pooled);
    try testing.expectEqual(@as(u64, 145 * page), p.decode_end);
    try testing.expect(p.decode_end <= p.bytes);
    try testing.expectEqual(@as(u32, 4), p.pooled);
    try testing.expectEqual(@as(u32, 2), p.fresh);
    try testing.expectEqual(@as(u64, 50 * page), p.fresh_bytes);
    // The same sizes give the same plan.
    var ext2 = [_]Item{ .{ .bytes = 50 * page }, .{ .bytes = 20 * page }, .{ .bytes = 50 * page }, .{ .bytes = 60 * page + 3 }, .{ .bytes = 2 * page }, .{ .bytes = 0 } };
    const p2 = try plan(&prompt, &window, &ext2, page, &order);
    try testing.expectEqual(p.decode_end, p2.decode_end);
    for (ext, ext2) |a, b| try testing.expect(a.pooled == b.pooled and a.offset == b.offset);
}

test "bank_pool: a window 0 past the prompt's bytes is refused" {
    var prompt = [_]Item{.{ .bytes = 4 * 16384 }};
    var window = [_]Item{.{ .bytes = 5 * 16384 }};
    var ext = [_]Item{};
    var order: [0]u32 = .{};
    try testing.expectError(error.BankPoolWindowTooLarge, plan(&prompt, &window, &ext, 16384, &order));
}
