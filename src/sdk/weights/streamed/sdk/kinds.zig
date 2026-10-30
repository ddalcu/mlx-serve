//! The `source` and `engine` kinds' registry tables (docs/plugins.md). Each carries the routing question; a kind's
//! full interface lands with its first consumer. A quant or an expert source is an arch's internal, bound at comptime.

const std = @import("std");
const peek = @import("peek.zig");
const check = @import("check.zig");

/// Opens a non-HF container: claims a model path before any file is read as a model directory.
pub const Source = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

    pub fn of(comptime T: type) Source {
        comptime {
            const w = "source " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
        }
        return .{ .name = T.name, .claims = T.claims };
    }
};

/// A whole engine behind an opaque session (ds4, llama.cpp): it gets none of the host's stack below HTTP.
pub const Engine = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,

    pub fn of(comptime T: type) Engine {
        comptime {
            const w = "engine " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
        }
        return .{ .name = T.name, .claims = T.claims };
    }
};

const testing = std.testing;

fn claimsModel(p: *const peek.ConfigPeek) ?peek.Priority {
    return if (p.modelType() != null) .generic else null;
}

test "sdk kinds: source and engine tables carry the name and the claim; a claim runs through the table" {
    const S = struct {
        pub const name = "fixture-source";
        pub const claims = claimsModel;
    };
    const src = comptime Source.of(S);
    const eng = comptime Engine.of(S);
    try testing.expectEqualStrings("fixture-source", src.name);
    try testing.expectEqualStrings("fixture-source", eng.name);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"llama\"}");
    const none = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{}");
    try testing.expectEqual(@as(?peek.Priority, .generic), src.claims(&p));
    try testing.expectEqual(@as(?peek.Priority, null), eng.claims(&none));
}

test "sdk kinds: an optional hook is present when declared and not `{}`" {
    const T = struct {
        pub const on = claimsModel;
        pub const off = {};
    };
    try testing.expect(check.has(T, "on") and !check.has(T, "off") and !check.has(T, "absent"));
}
