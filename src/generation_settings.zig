const std = @import("std");

pub const Surface = enum { chat, completions, messages, responses };
pub const Field = enum { temperature, top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty, max_tokens, enable_thinking, reasoning_effort, reasoning_budget };
pub const Effort = enum { none, minimal, low, medium, high, xhigh, max };
pub const Value = union(enum) { number: f64, integer: i64, boolean: bool, effort: Effort };
pub const Source = enum { client, model, global, cli, checkpoint, fallback };
pub const Rule = struct { value: Value, ignore_client: bool = false, source: Source = .global };
pub var cli_reasoning_budget: ?i32 = null;

pub const Profile = struct {
    rules: [std.enums.values(Field).len]?Rule = @splat(null),

    pub fn set(self: *Profile, field: Field, value: Value, ignore_client: bool) void {
        self.rules[@backingInt(field)] = .{ .value = value, .ignore_client = ignore_client };
    }
};

pub const Resolved = struct {
    profile: Profile = .{},
    values: [std.enums.values(Field).len]?std.json.Value = @splat(null),
    sources: [std.enums.values(Field).len]Source = @splat(.fallback),

    pub fn init(global: Profile, model: Profile) Resolved {
        var result = Resolved{ .profile = global };
        for (model.rules, 0..) |rule, i| if (rule) |r| {
            result.profile.rules[i] = r;
            result.profile.rules[i].?.source = .model;
        };
        return result;
    }

    pub fn locked(self: Resolved, field: Field) bool {
        const rule = self.profile.rules[@backingInt(field)] orelse return false;
        return rule.ignore_client;
    }

    pub fn resolveOptional(self: *Resolved, comptime T: type, field: Field, client: ?T, fallback: ?T, source: Source) ?T {
        const i = @backingInt(field);
        const rule = self.profile.rules[i];
        const value: ?T = if (rule != null and (rule.?.ignore_client or client == null)) blk: {
            self.sources[i] = rule.?.source;
            break :blk typedValue(T, rule.?.value);
        } else if (client) |v| blk: {
            self.sources[i] = .client;
            break :blk v;
        } else blk: {
            self.sources[i] = source;
            break :blk fallback;
        };
        self.values[i] = if (value) |v| diagnosticValue(T, v) else null;
        return value;
    }

    pub fn resolve(self: *Resolved, comptime T: type, field: Field, client: ?T, fallback: T, source: Source) T {
        return self.resolveOptional(T, field, client, fallback, source).?;
    }

    pub fn json(self: Resolved, allocator: std.mem.Allocator) ![]u8 {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var object: std.json.ObjectMap = .empty;
        for (std.enums.values(Field)) |field| {
            const i = @backingInt(field);
            const value = self.values[i] orelse continue;
            var item: std.json.ObjectMap = .empty;
            try item.put(a, "source", .{ .string = @tagName(self.sources[i]) });
            try item.put(a, "ignore_client", .{ .bool = self.locked(field) });
            try item.put(a, "value", value);
            try object.put(a, @tagName(field), .{ .object = item });
        }
        return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = object }, .{});
    }
};

fn typedValue(comptime T: type, value: Value) T {
    if (T == bool) return value.boolean;
    if (T == []const u8) return @tagName(value.effort);
    return switch (@typeInfo(T)) {
        .float => switch (value) {
            .number => |v| @floatCast(v),
            .integer => |v| @floatFromInt(v),
            else => unreachable,
        },
        .int => @intCast(value.integer),
        else => @compileError("unsupported generation value"),
    };
}

fn diagnosticValue(comptime T: type, value: T) std.json.Value {
    if (T == bool) return .{ .bool = value };
    if (T == []const u8) return .{ .string = value };
    return switch (@typeInfo(T)) {
        .float => .{ .float = value },
        .int => .{ .integer = value },
        else => @compileError("unsupported generation value"),
    };
}

pub fn validateThinkingFallback(result: Resolved, thinking: bool) !void {
    if (result.locked(.enable_thinking)) {
        const value = result.values[@backingInt(Field.enable_thinking)] orelse return;
        if (value.bool != thinking) return error.UnsupportedThinkingPolicy;
    }
    if (result.locked(.reasoning_effort)) {
        const value = result.values[@backingInt(Field.reasoning_effort)] orelse return;
        if (!std.mem.eql(u8, value.string, "none") and !thinking) return error.UnsupportedThinkingPolicy;
    }
}

pub fn validateEnginePolicy(result: Resolved, embedded: bool) !void {
    if (!embedded) return;
    for ([_]Field{ .repeat_penalty, .presence_penalty, .frequency_penalty }) |field| {
        const value = result.values[@backingInt(field)] orelse continue;
        const number: f64 = switch (value) {
            .float => |v| v,
            .integer => |v| @floatFromInt(v),
            else => continue,
        };
        const neutral: f64 = if (field == .repeat_penalty) 1 else 0;
        if (number != neutral and result.sources[@backingInt(field)] != .client) return error.UnsupportedEngineGenerationPolicy;
    }
}

pub fn parseProfile(value: std.json.Value) !Profile {
    if (value != .object) return error.InvalidGenerationSettings;
    var profile = Profile{};
    var it = value.object.iterator();
    while (it.next()) |item| {
        const field = std.meta.stringToEnum(Field, item.key_ptr.*) orelse return error.UnknownGenerationSetting;
        const object = item.value_ptr.*;
        if (object != .object) return error.InvalidGenerationSettings;
        for (object.object.keys()) |key| {
            if (!std.mem.eql(u8, key, "value") and !std.mem.eql(u8, key, "ignore_client")) return error.InvalidGenerationSettings;
        }
        const raw = object.object.get("value") orelse return error.InvalidGenerationSettings;
        const lock = object.object.get("ignore_client") orelse std.json.Value{ .bool = false };
        if (lock != .bool) return error.InvalidGenerationSettings;
        const v: Value = switch (field) {
            .enable_thinking => if (raw == .bool) .{ .boolean = raw.bool } else return error.InvalidGenerationSettings,
            .reasoning_effort => if (raw == .string) .{ .effort = std.meta.stringToEnum(Effort, raw.string) orelse return error.InvalidGenerationSettings } else return error.InvalidGenerationSettings,
            .top_k, .max_tokens, .reasoning_budget => blk: {
                const number: f64 = switch (raw) {
                    .integer => |n| @floatFromInt(n),
                    .float => |n| n,
                    else => return error.InvalidGenerationSettings,
                };
                const min: f64 = if (field == .reasoning_budget) -1 else 0;
                const max: f64 = if (field == .top_k) 1000 else std.math.maxInt(i32);
                if (!std.math.isFinite(number) or number < min or number > max or @trunc(number) != number) return error.InvalidGenerationSettings;
                break :blk .{ .integer = @intFromFloat(number) };
            },
            else => blk: {
                const number: f64 = switch (raw) {
                    .integer => |n| @floatFromInt(n),
                    .float => |n| n,
                    else => return error.InvalidGenerationSettings,
                };
                const max: f64 = if (field == .repeat_penalty) 10 else if (field == .top_p) 1 else 2;
                const min: f64 = if (field == .repeat_penalty) 0.01 else 0;
                if (!std.math.isFinite(number) or number < min or number > max) return error.InvalidGenerationSettings;
                break :blk .{ .number = number };
            },
        };
        profile.set(field, v, lock.bool);
    }
    return profile;
}

pub fn profileJson(a: std.mem.Allocator, profile: Profile) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const alloc = arena.allocator();
    var object: std.json.ObjectMap = .empty;
    for (std.enums.values(Field)) |field| if (profile.rules[@backingInt(field)]) |rule| {
        var item: std.json.ObjectMap = .empty;
        const value: std.json.Value = switch (rule.value) {
            .number => |v| .{ .float = v },
            .integer => |v| .{ .integer = v },
            .boolean => |v| .{ .bool = v },
            .effort => |v| .{ .string = @tagName(v) },
        };
        try item.put(alloc, "value", value);
        try item.put(alloc, "ignore_client", .{ .bool = rule.ignore_client });
        try object.put(alloc, @tagName(field), .{ .object = item });
    };
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = object }, .{});
}

test "generation settings: typed omissions retain defaults and model rules replace global locks" {
    var global = Profile{};
    global.set(.temperature, .{ .number = 0.25 }, false);
    global.set(.max_tokens, .{ .integer = 100 }, true);
    var model = Profile{};
    model.set(.max_tokens, .{ .integer = 40 }, false);
    var result = Resolved.init(global, model);
    try std.testing.expectEqual(@as(f32, 0.25), result.resolve(f32, .temperature, null, 1, .fallback));
    try std.testing.expectEqual(@as(u32, 12), result.resolve(u32, .max_tokens, 12, 200, .fallback));
    try std.testing.expect(!result.locked(.max_tokens));
    model.set(.max_tokens, .{ .integer = 40 }, true);
    result = Resolved.init(global, model);
    try std.testing.expectEqual(@as(u32, 40), result.resolve(u32, .max_tokens, 12, 200, .fallback));
    try std.testing.expectEqual(Source.model, result.sources[@backingInt(Field.max_tokens)]);
}

test "generation settings: every field has client model global precedence and independent locks" {
    inline for (std.enums.values(Field)) |field| {
        const T = switch (field) {
            .enable_thinking => bool,
            .reasoning_effort => []const u8,
            .top_k, .max_tokens, .reasoning_budget => i32,
            else => f32,
        };
        const values: [3]Value = switch (field) {
            .enable_thinking => .{ .{ .boolean = true }, .{ .boolean = false }, .{ .boolean = true } },
            .reasoning_effort => .{ .{ .effort = .low }, .{ .effort = .medium }, .{ .effort = .high } },
            .top_k, .max_tokens, .reasoning_budget => .{ .{ .integer = 1 }, .{ .integer = 2 }, .{ .integer = 3 } },
            else => .{ .{ .number = 1 }, .{ .number = 2 }, .{ .number = 3 } },
        };
        var global = Profile{};
        var model = Profile{};
        global.set(field, values[0], true);
        model.set(field, values[1], false);
        var result = Resolved.init(global, model);
        try std.testing.expectEqualDeep(typedValue(T, values[2]), result.resolve(T, field, typedValue(T, values[2]), typedValue(T, values[0]), .fallback));
        try std.testing.expect(!result.locked(field));
        model.set(field, values[1], true);
        result = Resolved.init(global, model);
        try std.testing.expectEqualDeep(typedValue(T, values[1]), result.resolve(T, field, typedValue(T, values[2]), typedValue(T, values[0]), .fallback));
        try std.testing.expect(result.locked(field));
        result = Resolved.init(global, .{});
        try std.testing.expectEqualDeep(typedValue(T, values[0]), result.resolve(T, field, typedValue(T, values[2]), typedValue(T, values[1]), .fallback));
    }
}

test "generation settings: explicit neutral values survive unlocked defaults" {
    var profile = Profile{};
    profile.set(.top_k, .{ .integer = 40 }, false);
    profile.set(.repeat_penalty, .{ .number = 1.5 }, false);
    profile.set(.presence_penalty, .{ .number = 1 }, false);
    profile.set(.enable_thinking, .{ .boolean = true }, false);
    profile.set(.reasoning_budget, .{ .integer = 1024 }, false);
    var result = Resolved.init(profile, .{});
    try std.testing.expectEqual(@as(u32, 0), result.resolve(u32, .top_k, 0, 20, .checkpoint));
    try std.testing.expectEqual(@as(f32, 1), result.resolve(f32, .repeat_penalty, 1, 1.1, .checkpoint));
    try std.testing.expectEqual(@as(f32, 0), result.resolve(f32, .presence_penalty, 0, 0.5, .checkpoint));
    try std.testing.expect(!result.resolve(bool, .enable_thinking, false, true, .checkpoint));
    try std.testing.expectEqual(@as(i32, -1), result.resolve(i32, .reasoning_budget, -1, 2048, .fallback));
}

test "generation settings: invalid configured locks never fall open" {
    for ([_][]const u8{ "{\"top_k\":{\"value\":-1}}", "{\"enable_thinking\":{\"value\":false,\"ignore_client\":1}}", "{\"reasoning_budget\":{\"value\":-2}}", "{\"reasoning_effort\":{\"value\":\"banana\"}}", "{\"top_k\":{\"value\":0,\"ignore_clent\":true}}" }) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.InvalidGenerationSettings, parseProfile(parsed.value));
    }
}

test "generation settings: configured integer wire values and diagnostics round trip" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"reasoning_budget\":{\"value\":1024.0,\"ignore_client\":true}}", .{});
    defer parsed.deinit();
    var result = Resolved.init(try parseProfile(parsed.value), .{});
    try std.testing.expectEqual(@as(i32, 1024), result.resolve(i32, .reasoning_budget, 4096, -1, .fallback));
    const text = try result.json(a);
    defer a.free(text);
    const diagnostics = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer diagnostics.deinit();
    const budget = diagnostics.value.object.get("reasoning_budget").?.object;
    try std.testing.expectEqualStrings("global", budget.get("source").?.string);
    try std.testing.expect(budget.get("ignore_client").?.bool);
    const profile = try profileJson(a, result.profile);
    defer a.free(profile);
    const roundtrip = try std.json.parseFromSlice(std.json.Value, a, profile, .{});
    defer roundtrip.deinit();
    try std.testing.expectEqualDeep(result.profile, try parseProfile(roundtrip.value));
}

test "generation settings: unsupported configured penalties and thinking never pretend to work" {
    var profile = Profile{};
    profile.set(.repeat_penalty, .{ .number = 1.5 }, false);
    profile.set(.enable_thinking, .{ .boolean = true }, true);
    var result = Resolved.init(profile, .{});
    _ = result.resolve(f32, .repeat_penalty, null, 1, .fallback);
    _ = result.resolve(bool, .enable_thinking, false, false, .fallback);
    try std.testing.expectError(error.UnsupportedEngineGenerationPolicy, validateEnginePolicy(result, true));
    try std.testing.expectError(error.UnsupportedThinkingPolicy, validateThinkingFallback(result, false));
    _ = result.resolve(f32, .repeat_penalty, 1, 1, .fallback);
    try validateEnginePolicy(result, true);
    try validateThinkingFallback(result, true);
}
