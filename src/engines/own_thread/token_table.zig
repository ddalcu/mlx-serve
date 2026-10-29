//! Each token's bytes, decoded once on first ask and kept for the engine's lifetime: `Engine.token_bytes` for a
//! GGUF engine whose own lookup hands back a fresh copy per call (ds4, llama.cpp). Safe from any thread.

const std = @import("std");
const io = @import("book.zig").io;

pub const TokenTable = struct {
    mu: std.Io.Mutex = .init,
    pieces: []?[]u8,

    pub fn init(gpa: std.mem.Allocator, vocab: usize) !TokenTable {
        const pieces = try gpa.alloc(?[]u8, vocab);
        @memset(pieces, null);
        return .{ .pieces = pieces };
    }

    pub fn deinit(t: *TokenTable, gpa: std.mem.Allocator) void {
        for (t.pieces) |p| if (p) |b| gpa.free(b);
        gpa.free(t.pieces);
    }

    /// `id`'s bytes via `engine.detokenizeOne`; empty for an id past the vocabulary or a failed decode.
    pub fn get(t: *TokenTable, gpa: std.mem.Allocator, engine: anytype, id: u32) []const u8 {
        if (id >= t.pieces.len) return "";
        t.mu.lockUncancelable(io);
        defer t.mu.unlock(io);
        if (t.pieces[id]) |b| return b;
        const b = engine.detokenizeOne(gpa, @intCast(id)) catch return "";
        t.pieces[id] = b;
        return b;
    }
};

// ── tests ──

test "token table: decodes each id once, keeps the bytes" {
    const gpa = std.testing.allocator;
    const Echo = struct {
        calls: usize = 0,
        fn detokenizeOne(e: *@This(), a: std.mem.Allocator, id: i32) ![]u8 {
            e.calls += 1;
            return std.fmt.allocPrint(a, "<{d}>", .{id});
        }
    };
    var echo: Echo = .{};
    var t = try TokenTable.init(gpa, 4);
    defer t.deinit(gpa);
    const a = t.get(gpa, &echo, 2);
    try std.testing.expectEqualStrings("<2>", a);
    try std.testing.expectEqual(a.ptr, t.get(gpa, &echo, 2).ptr);
    try std.testing.expectEqual(@as(usize, 1), echo.calls);
    try std.testing.expectEqualStrings("", t.get(gpa, &echo, 9));
}
