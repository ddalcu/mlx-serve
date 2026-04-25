//! JSON Schema → internal IR for grammar-constrained decoding.
//!
//! Supports the "common" subset:
//!   * type (single string or array, incl. null/integer/number/string/boolean/object/array)
//!   * properties / required / additionalProperties
//!   * items (homogeneous array)
//!   * anyOf / oneOf (treated identically for grammar purposes)
//!   * enum / const (treated as a 1-element enum)
//!   * minLength / maxLength
//!   * minItems / maxItems
//!   * minimum / maximum / exclusiveMinimum / exclusiveMaximum
//!   * pattern (regex; see regex.zig)
//!
//! Strict OpenAI mode: when `additionalProperties` is unspecified, defaults to `false`.
//!
//! A `Schema` owns an arena allocator; freeing the schema frees every node, string,
//! property name, and compiled regex in one shot.

const std = @import("std");
const regex = @import("regex.zig");

pub const Kind = enum {
    any,
    null_,
    boolean,
    integer,
    number,
    string,
    array,
    object,
    enum_,
    any_of,
};

pub const Property = struct {
    name: []const u8,
    schema: *const Node,
    required: bool,
};

pub const Node = struct {
    kind: Kind,

    // string
    str_min_len: ?u32 = null,
    str_max_len: ?u32 = null,
    str_pattern: ?*const regex.Nfa = null,

    // number / integer
    num_min: ?f64 = null,
    num_max: ?f64 = null,
    num_excl_min: ?f64 = null,
    num_excl_max: ?f64 = null,

    // array
    arr_items: ?*const Node = null,
    arr_min_items: ?u32 = null,
    arr_max_items: ?u32 = null,

    // object
    obj_properties: []const Property = &.{}, // sorted alphabetically by name
    obj_additional: bool = false,

    // enum / const
    enum_values: []const EnumValue = &.{},

    // any_of (and one_of)
    any_of_options: []const *const Node = &.{},
};

/// A single enum/const value rendered as canonical JSON text.
/// Keeping the textual form lets the grammar prefix-match incrementally.
pub const EnumValue = struct {
    /// Canonical JSON encoding, e.g. `"text"`, `42`, `true`, `null`, `{"a":1}`.
    json: []const u8,
};

pub const Schema = struct {
    arena: std.heap.ArenaAllocator,
    root: *const Node,

    pub fn deinit(self: *Schema) void {
        self.arena.deinit();
    }
};

pub const ParseError = error{
    InvalidSchema,
    UnsupportedConstruct,
    OutOfMemory,
    InvalidPattern,
};

/// Parse a `std.json.Value` into a `Schema`. The returned schema owns its memory.
pub fn parse(gpa: std.mem.Allocator, value: std.json.Value) ParseError!Schema {
    var schema: Schema = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .root = undefined,
    };
    errdefer schema.arena.deinit();

    const arena = schema.arena.allocator();
    schema.root = try parseNode(arena, value);
    return schema;
}

fn parseNode(arena: std.mem.Allocator, value: std.json.Value) ParseError!*const Node {
    if (value != .object) {
        // Boolean schema: `true` (any), `false` (none — we don't model "none").
        if (value == .bool) {
            const node = try arena.create(Node);
            node.* = .{ .kind = .any };
            return node;
        }
        return error.InvalidSchema;
    }
    const obj = value.object;

    // anyOf / oneOf — branching schemas. Both compile to a Kind.any_of node.
    if (obj.get("anyOf")) |v| return try parseAnyOf(arena, v);
    if (obj.get("oneOf")) |v| return try parseAnyOf(arena, v);

    // enum / const — one or more allowed literal values.
    if (obj.get("const")) |v| {
        const node = try arena.create(Node);
        node.* = .{
            .kind = .enum_,
            .enum_values = try makeEnumValues(arena, &.{v}),
        };
        return node;
    }
    if (obj.get("enum")) |v| {
        if (v != .array) return error.InvalidSchema;
        const node = try arena.create(Node);
        node.* = .{
            .kind = .enum_,
            .enum_values = try makeEnumValues(arena, v.array.items),
        };
        return node;
    }

    // type — may be a string or an array of strings. Array form compiles to anyOf.
    const type_val = obj.get("type") orelse {
        // Unspecified type — treat as `any`. Consumers will allow any JSON value.
        const node = try arena.create(Node);
        node.* = .{ .kind = .any };
        return node;
    };

    if (type_val == .array) {
        var options = try arena.alloc(*const Node, type_val.array.items.len);
        for (type_val.array.items, 0..) |t, i| {
            if (t != .string) return error.InvalidSchema;
            // Synthesize a single-type schema by cloning `obj` minus the `type` array.
            options[i] = try parseSingleType(arena, obj, t.string);
        }
        const node = try arena.create(Node);
        node.* = .{ .kind = .any_of, .any_of_options = options };
        return node;
    }
    if (type_val != .string) return error.InvalidSchema;
    return try parseSingleType(arena, obj, type_val.string);
}

fn parseAnyOf(arena: std.mem.Allocator, v: std.json.Value) ParseError!*const Node {
    if (v != .array) return error.InvalidSchema;
    if (v.array.items.len == 0) return error.InvalidSchema;
    var options = try arena.alloc(*const Node, v.array.items.len);
    for (v.array.items, 0..) |item, i| {
        options[i] = try parseNode(arena, item);
    }
    const node = try arena.create(Node);
    node.* = .{ .kind = .any_of, .any_of_options = options };
    return node;
}

fn parseSingleType(arena: std.mem.Allocator, obj: std.json.ObjectMap, t: []const u8) ParseError!*const Node {
    const node = try arena.create(Node);

    if (std.mem.eql(u8, t, "null")) {
        node.* = .{ .kind = .null_ };
    } else if (std.mem.eql(u8, t, "boolean")) {
        node.* = .{ .kind = .boolean };
    } else if (std.mem.eql(u8, t, "integer")) {
        node.* = .{
            .kind = .integer,
            .num_min = readF64(obj, "minimum"),
            .num_max = readF64(obj, "maximum"),
            .num_excl_min = readF64(obj, "exclusiveMinimum"),
            .num_excl_max = readF64(obj, "exclusiveMaximum"),
        };
    } else if (std.mem.eql(u8, t, "number")) {
        node.* = .{
            .kind = .number,
            .num_min = readF64(obj, "minimum"),
            .num_max = readF64(obj, "maximum"),
            .num_excl_min = readF64(obj, "exclusiveMinimum"),
            .num_excl_max = readF64(obj, "exclusiveMaximum"),
        };
    } else if (std.mem.eql(u8, t, "string")) {
        var pattern_nfa: ?*const regex.Nfa = null;
        if (obj.get("pattern")) |pv| {
            if (pv != .string) return error.InvalidSchema;
            pattern_nfa = regex.compile(arena, pv.string) catch return error.InvalidPattern;
        }
        node.* = .{
            .kind = .string,
            .str_min_len = readU32(obj, "minLength"),
            .str_max_len = readU32(obj, "maxLength"),
            .str_pattern = pattern_nfa,
        };
    } else if (std.mem.eql(u8, t, "array")) {
        var items_node: ?*const Node = null;
        if (obj.get("items")) |iv| {
            // Tuple form (`items: [a, b]`) is not in our subset.
            if (iv == .array) return error.UnsupportedConstruct;
            items_node = try parseNode(arena, iv);
        }
        node.* = .{
            .kind = .array,
            .arr_items = items_node,
            .arr_min_items = readU32(obj, "minItems"),
            .arr_max_items = readU32(obj, "maxItems"),
        };
    } else if (std.mem.eql(u8, t, "object")) {
        // additionalProperties defaults to `false` (OpenAI strict semantics).
        var additional: bool = false;
        if (obj.get("additionalProperties")) |ap| {
            if (ap == .bool) additional = ap.bool
            // `additionalProperties: <schema>` would constrain extra keys; we don't
            // model that yet, so treat any non-bool form as "allow any extra".
            else additional = true;
        }
        const props = if (obj.get("properties")) |pv|
            try parseProperties(arena, pv, obj.get("required"))
        else
            &[_]Property{};
        node.* = .{
            .kind = .object,
            .obj_properties = props,
            .obj_additional = additional,
        };
    } else {
        return error.UnsupportedConstruct;
    }

    return node;
}

fn parseProperties(
    arena: std.mem.Allocator,
    properties: std.json.Value,
    required: ?std.json.Value,
) ParseError![]Property {
    if (properties != .object) return error.InvalidSchema;
    const map = properties.object;

    // Build required set
    var req_set: std.StringHashMapUnmanaged(void) = .empty;
    if (required) |r| {
        if (r != .array) return error.InvalidSchema;
        for (r.array.items) |item| {
            if (item != .string) return error.InvalidSchema;
            try req_set.put(arena, item.string, {});
        }
    }

    var props = try arena.alloc(Property, map.count());
    var i: usize = 0;
    var it = map.iterator();
    while (it.next()) |entry| : (i += 1) {
        const name_owned = try arena.dupe(u8, entry.key_ptr.*);
        const child = try parseNode(arena, entry.value_ptr.*);
        props[i] = .{
            .name = name_owned,
            .schema = child,
            .required = req_set.contains(name_owned),
        };
    }
    // Sort alphabetically — keeps prefix-matching deterministic regardless of
    // the JSON parser's internal map order.
    std.mem.sort(Property, props, {}, lessThanProperty);
    return props;
}

fn lessThanProperty(_: void, a: Property, b: Property) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn makeEnumValues(arena: std.mem.Allocator, items: []const std.json.Value) ParseError![]EnumValue {
    var out = try arena.alloc(EnumValue, items.len);
    for (items, 0..) |v, i| {
        out[i] = .{ .json = try canonicalJson(arena, v) };
    }
    return out;
}

/// Render a `std.json.Value` to canonical JSON text (no extraneous whitespace).
/// Used so the grammar can prefix-match against allowed enum/const literals byte-by-byte.
fn canonicalJson(arena: std.mem.Allocator, value: std.json.Value) ParseError![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(arena);
    try writeCanonical(arena, &buf, value);
    return try buf.toOwnedSlice(arena);
}

fn writeCanonical(arena: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), value: std.json.Value) ParseError!void {
    switch (value) {
        .null => try buf.appendSlice(arena, "null"),
        .bool => |b| try buf.appendSlice(arena, if (b) "true" else "false"),
        .integer => |i| {
            try std.fmt.format(buf.writer(arena), "{d}", .{i});
        },
        .number_string => |s| try buf.appendSlice(arena, s),
        .float => |f| {
            try std.fmt.format(buf.writer(arena), "{d}", .{f});
        },
        .string => |s| try writeJsonString(arena, buf, s),
        .array => |arr| {
            try buf.append(arena, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try buf.append(arena, ',');
                try writeCanonical(arena, buf, item);
            }
            try buf.append(arena, ']');
        },
        .object => |o| {
            try buf.append(arena, '{');
            // Sorted keys for deterministic output
            var keys = arena.alloc([]const u8, o.count()) catch return error.OutOfMemory;
            defer arena.free(keys);
            var it = o.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) keys[i] = entry.key_ptr.*;
            std.mem.sort([]const u8, keys, {}, struct {
                fn lt(_: void, a: []const u8, b: []const u8) bool {
                    return std.mem.lessThan(u8, a, b);
                }
            }.lt);
            for (keys, 0..) |k, j| {
                if (j > 0) try buf.append(arena, ',');
                try writeJsonString(arena, buf, k);
                try buf.append(arena, ':');
                try writeCanonical(arena, buf, o.get(k).?);
            }
            try buf.append(arena, '}');
        },
    }
}

fn writeJsonString(arena: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) ParseError!void {
    try buf.append(arena, '"');
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(arena, "\\\""),
            '\\' => try buf.appendSlice(arena, "\\\\"),
            '\n' => try buf.appendSlice(arena, "\\n"),
            '\r' => try buf.appendSlice(arena, "\\r"),
            '\t' => try buf.appendSlice(arena, "\\t"),
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(arena, n);
            },
            else => try buf.append(arena, c),
        }
    }
    try buf.append(arena, '"');
}

fn readU32(obj: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i < 0) null else @intCast(i),
        else => null,
    };
}

fn readF64(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn parseStr(gpa: std.mem.Allocator, src: []const u8) !Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return parse(gpa, v.value);
}

test "parse simple object schema with required" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.object, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.obj_properties.len);
    try testing.expectEqualStrings("name", s.root.obj_properties[0].name);
    try testing.expect(s.root.obj_properties[0].required);
    try testing.expectEqual(Kind.string, s.root.obj_properties[0].schema.kind);
    try testing.expect(!s.root.obj_additional);
}

test "additionalProperties defaults to false (OpenAI strict)" {
    var s = try parseStr(testing.allocator, "{\"type\":\"object\"}");
    defer s.deinit();
    try testing.expect(!s.root.obj_additional);
}

test "additionalProperties:true is honored" {
    var s = try parseStr(testing.allocator, "{\"type\":\"object\",\"additionalProperties\":true}");
    defer s.deinit();
    try testing.expect(s.root.obj_additional);
}

test "anyOf becomes any_of node" {
    var s = try parseStr(testing.allocator,
        \\{"anyOf":[{"type":"string"},{"type":"null"}]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.any_of, s.root.kind);
    try testing.expectEqual(@as(usize, 2), s.root.any_of_options.len);
    try testing.expectEqual(Kind.string, s.root.any_of_options[0].kind);
    try testing.expectEqual(Kind.null_, s.root.any_of_options[1].kind);
}

test "type as array compiles to any_of" {
    var s = try parseStr(testing.allocator,
        \\{"type":["string","null"]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.any_of, s.root.kind);
    try testing.expectEqual(@as(usize, 2), s.root.any_of_options.len);
}

test "enum values render to canonical JSON" {
    var s = try parseStr(testing.allocator,
        \\{"enum":["text","cruise_cards",42,true,null]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.enum_, s.root.kind);
    try testing.expectEqual(@as(usize, 5), s.root.enum_values.len);
    try testing.expectEqualStrings("\"text\"", s.root.enum_values[0].json);
    try testing.expectEqualStrings("\"cruise_cards\"", s.root.enum_values[1].json);
    try testing.expectEqualStrings("42", s.root.enum_values[2].json);
    try testing.expectEqualStrings("true", s.root.enum_values[3].json);
    try testing.expectEqualStrings("null", s.root.enum_values[4].json);
}

test "const collapses to single-element enum" {
    var s = try parseStr(testing.allocator, "{\"const\":\"hello\"}");
    defer s.deinit();
    try testing.expectEqual(Kind.enum_, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.enum_values.len);
    try testing.expectEqualStrings("\"hello\"", s.root.enum_values[0].json);
}

test "string constraints are captured" {
    var s = try parseStr(testing.allocator,
        \\{"type":"string","minLength":2,"maxLength":10}
    );
    defer s.deinit();
    try testing.expectEqual(@as(?u32, 2), s.root.str_min_len);
    try testing.expectEqual(@as(?u32, 10), s.root.str_max_len);
}

test "number constraints are captured" {
    var s = try parseStr(testing.allocator,
        \\{"type":"number","minimum":0,"exclusiveMaximum":1}
    );
    defer s.deinit();
    try testing.expectEqual(@as(?f64, 0), s.root.num_min);
    try testing.expectEqual(@as(?f64, 1), s.root.num_excl_max);
}

test "array with items schema" {
    var s = try parseStr(testing.allocator,
        \\{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.array, s.root.kind);
    try testing.expect(s.root.arr_items != null);
    try testing.expectEqual(Kind.integer, s.root.arr_items.?.kind);
    try testing.expectEqual(@as(?u32, 1), s.root.arr_min_items);
    try testing.expectEqual(@as(?u32, 3), s.root.arr_max_items);
}

test "properties are sorted alphabetically" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"},"m":{"type":"boolean"}}}
    );
    defer s.deinit();
    try testing.expectEqual(@as(usize, 3), s.root.obj_properties.len);
    try testing.expectEqualStrings("a", s.root.obj_properties[0].name);
    try testing.expectEqualStrings("m", s.root.obj_properties[1].name);
    try testing.expectEqualStrings("z", s.root.obj_properties[2].name);
}

test "nested cruise-app-style schema" {
    const src =
        \\{
        \\  "type":"object",
        \\  "properties":{
        \\    "blocks":{
        \\      "type":"array",
        \\      "items":{
        \\        "anyOf":[
        \\          {"type":"object","properties":{"type":{"enum":["text"]},"content":{"type":"string"}},"required":["type","content"],"additionalProperties":false}
        \\        ]
        \\      }
        \\    }
        \\  },
        \\  "required":["blocks"],
        \\  "additionalProperties":false
        \\}
    ;
    var s = try parseStr(testing.allocator, src);
    defer s.deinit();
    try testing.expectEqual(Kind.object, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.obj_properties.len);
    const blocks = s.root.obj_properties[0].schema;
    try testing.expectEqual(Kind.array, blocks.kind);
    const item = blocks.arr_items.?;
    try testing.expectEqual(Kind.any_of, item.kind);
    try testing.expectEqual(@as(usize, 1), item.any_of_options.len);
    const text_opt = item.any_of_options[0];
    try testing.expectEqual(Kind.object, text_opt.kind);
    // Type discriminator should be an enum
    var found_type = false;
    for (text_opt.obj_properties) |p| {
        if (std.mem.eql(u8, p.name, "type")) {
            found_type = true;
            try testing.expectEqual(Kind.enum_, p.schema.kind);
            try testing.expectEqualStrings("\"text\"", p.schema.enum_values[0].json);
        }
    }
    try testing.expect(found_type);
}

test "boolean schema true => any" {
    const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, "true", .{});
    defer v.deinit();
    var s = try parse(testing.allocator, v.value);
    defer s.deinit();
    try testing.expectEqual(Kind.any, s.root.kind);
}

test "unsupported tuple-form items returns error" {
    const result = parseStr(testing.allocator,
        \\{"type":"array","items":[{"type":"string"},{"type":"number"}]}
    );
    try testing.expectError(error.UnsupportedConstruct, result);
}
