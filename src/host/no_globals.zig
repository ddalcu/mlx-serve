//! The no-globals check (plan 3.10: no mutable global in `sdk/` or `src/plugins/`): every container-level `var`
//! (a struct's, a nested struct's, one inside a function's local type) and every `threadlocal var`, read from Zig's
//! own parser. Report-only until P3 removes them per plugin:
//!   zig run src/host/no_globals.zig -- src/sdk src/plugins
//! prints one `path:line: name` per global, then `folder vars threadlocal` per first-level folder.

const std = @import("std");
const Ast = std.zig.Ast;

pub const Global = struct {
    line: u32,
    name: []const u8,
    thread_local: bool,
};

/// The mutable globals in one file, in source order; `name` borrows `source`. Null on a parse error.
pub fn scan(gpa: std.mem.Allocator, source: [:0]const u8) !?[]Global {
    var tree = try Ast.parse(gpa, source, .{});
    defer tree.deinit(gpa);
    if (tree.errors.len != 0) return null;
    var out: std.ArrayList(Global) = .empty;
    errdefer out.deinit(gpa);
    var buf: [2]Ast.Node.Index = undefined;
    for (0..tree.nodes.len) |i| {
        const node: Ast.Node.Index = @fromBackingInt(@intCast(i));
        const container = tree.fullContainerDecl(&buf, node) orelse continue;
        for (container.ast.members) |m| {
            const v = tree.fullVarDecl(m) orelse continue;
            if (tree.tokenTag(v.ast.mut_token) != .keyword_var) continue;
            try out.append(gpa, .{
                .line = @intCast(tree.tokenLocation(0, v.ast.mut_token).line + 1),
                .name = tree.tokenSlice(v.ast.mut_token + 1),
                .thread_local = v.threadlocal_token != null,
            });
        }
    }
    std.mem.sort(Global, out.items, {}, struct {
        fn lt(_: void, a: Global, b: Global) bool {
            return a.line < b.line;
        }
    }.lt);
    return try out.toOwnedSlice(gpa);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const roots: []const []const u8 = if (args.len > 1) args[1..] else &.{ "src/sdk", "src/plugins" };

    var out_buf: [8192]u8 = undefined;
    var out_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &out_w.interface;

    const Tally = struct { vars: usize = 0, tls: usize = 0 };
    var folders: std.StringArrayHashMapUnmanaged(Tally) = .empty;
    defer folders.deinit(gpa);
    const arena = init.arena.allocator();

    for (roots) |root| {
        var dir = try std.Io.Dir.cwd().openDir(io, root, .{ .iterate = true });
        defer dir.close(io);
        var walker = try dir.walk(gpa);
        defer walker.deinit();
        var files: std.ArrayList([]const u8) = .empty;
        while (try walker.next(io)) |e| {
            if (e.kind == .file and std.mem.endsWith(u8, e.path, ".zig")) try files.append(arena, try arena.dupe(u8, e.path));
        }
        std.mem.sort([]const u8, files.items, {}, struct {
            fn lt(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.lt);
        for (files.items) |rel| {
            const src = try dir.readFileAllocOptions(io, rel, arena, .limited(64 << 20), .of(u8), 0);
            const globals = try scan(gpa, src) orelse {
                try w.print("{s}/{s}: parse error, not scanned\n", .{ root, rel });
                continue;
            };
            defer gpa.free(globals);
            const first = if (std.mem.indexOfAny(u8, rel, "/\\")) |k| rel[0..k] else rel;
            const folder = try std.fmt.allocPrint(arena, "{s}/{s}", .{ root, first });
            const t = try folders.getOrPut(gpa, folder);
            if (!t.found_existing) t.value_ptr.* = .{};
            for (globals) |g| {
                try w.print("{s}/{s}:{d}: {s}{s}\n", .{ root, rel, g.line, if (g.thread_local) "threadlocal " else "", g.name });
                if (g.thread_local) t.value_ptr.tls += 1 else t.value_ptr.vars += 1;
            }
        }
    }
    var total: Tally = .{};
    try w.print("\nfolder vars threadlocal\n", .{});
    for (folders.keys(), folders.values()) |k, v| {
        if (v.vars + v.tls == 0) continue;
        try w.print("{s} {d} {d}\n", .{ k, v.vars, v.tls });
        total.vars += v.vars;
        total.tls += v.tls;
    }
    try w.print("total {d} {d}\n", .{ total.vars, total.tls });
    try w.flush();
}

test "no-globals scan: container-level and threadlocal vars at any nesting, never locals or consts" {
    const src =
        \\const std = @import("std");
        \\var a: u32 = 0;
        \\pub var b: ?*u8 = null;
        \\threadlocal var c: u8 = 0;
        \\const d = 1;
        \\const S = struct {
        \\    var e: u8 = 0;
        \\    x: u8,
        \\    fn f() void {
        \\        var local: u8 = 0;
        \\        local += 1;
        \\        const Inner = struct { var g: u8 = 0; };
        \\        _ = Inner;
        \\    }
        \\};
        \\const U = union(enum) { p: u8, pub var h: u8 = 0; };
        \\test { var t: u8 = 0; _ = &t; }
    ;
    const got = (try scan(std.testing.allocator, src)).?;
    defer std.testing.allocator.free(got);
    const want = [_]struct { u32, []const u8, bool }{
        .{ 2, "a", false }, .{ 3, "b", false }, .{ 4, "c", true }, .{ 7, "e", false }, .{ 12, "g", false }, .{ 16, "h", false },
    };
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| {
        try std.testing.expectEqual(w[0], g.line);
        try std.testing.expectEqualStrings(w[1], g.name);
        try std.testing.expectEqual(w[2], g.thread_local);
    }
    try std.testing.expectEqual(@as(?[]Global, null), try scan(std.testing.allocator, "var = ;"));
}
