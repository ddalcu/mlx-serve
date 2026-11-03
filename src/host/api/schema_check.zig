//! A request whose JSON schema the grammar cannot enforce is a 400 on every surface, before any tokenization.

const std = @import("std");
const json_schema = @import("../../sdk/grammar/json_schema.zig");

/// The 400 message for a schema `json_schema.parse` rejects, null when it parses.
pub fn grammarSchemaError(allocator: std.mem.Allocator, schema: std.json.Value) ?[]const u8 {
    var parsed = json_schema.parse(allocator, schema) catch |err| return switch (err) {
        error.UnsupportedConstruct => "JSON schema cannot be enforced (UnsupportedConstruct): an unknown type, tuple items or another construct the grammar does not support",
        error.InvalidPattern => "JSON schema cannot be enforced (InvalidPattern): a pattern the grammar's regex engine does not compile",
        error.InvalidSchema => "JSON schema is invalid (InvalidSchema)",
        error.OutOfMemory => null,
    };
    parsed.deinit();
    return null;
}

fn check(src: []const u8) !?[]const u8 {
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, src, .{});
    defer parsed.deinit();
    return grammarSchemaError(std.testing.allocator, parsed.value);
}

test "schema check: schemas the grammar rejects get a named message, valid ones pass" {
    try std.testing.expect((try check("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"strng\"}},\"required\":[\"a\"]}")) != null);
    try std.testing.expect((try check("{\"type\":\"array\",\"items\":[{\"type\":\"string\"},{\"type\":\"integer\"}]}")) != null);
    try std.testing.expect((try check("{\"type\":\"string\",\"pattern\":\"(?<=x)y\"}")) != null);
    try std.testing.expect((try check("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},\"required\":\"a\"}")) != null);
    try std.testing.expect((try check("{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\"}},\"required\":[\"a\"]}")) == null);
}
