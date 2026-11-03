//! Generation defaults: per-field rules in `~/.mlx-serve/generation-settings.json` (global) and
//! under a model's `generation_defaults` in `model-settings.json`. A rule fills a field the client
//! omitted; `"ignore_client": true` makes it replace the client's value.
const std = @import("std");

pub const Field = enum { temperature, top_p, top_k, min_p, repeat_penalty, presence_penalty, frequency_penalty, max_tokens, enable_thinking, reasoning_effort, reasoning_budget };
pub const Effort = enum { none, minimal, low, medium, high, xhigh, max };
pub const Value = union(enum) { number: f64, boolean: bool, effort: Effort };
pub const Rule = struct { value: Value, force: bool = false };

pub const Profile = struct {
    rules: std.EnumArray(Field, ?Rule) = .initFill(null),

    pub fn set(self: *Profile, field: Field, v: Value, force: bool) void {
        self.rules.set(field, .{ .value = v, .force = force });
    }

    /// The model's rules replace the global ones field by field, force flag included.
    pub fn overlay(global: Profile, model: Profile) Profile {
        var out = global;
        for (std.enums.values(Field)) |f| if (model.rules.get(f)) |r| out.rules.set(f, r);
        return out;
    }

    pub fn forced(self: Profile, field: Field) bool {
        return if (self.rules.get(field)) |r| r.force else false;
    }

    pub fn value(self: Profile, comptime T: type, field: Field) ?T {
        const r = self.rules.get(field) orelse return null;
        return switch (T) {
            bool => r.value.boolean,
            Effort => r.value.effort,
            else => switch (@typeInfo(T)) {
                .float => @floatCast(r.value.number),
                .int => @intFromFloat(r.value.number),
                else => @compileError("unsupported generation value"),
            },
        };
    }

    /// A forced rule, else the client's value, else the rule, else `fallback`.
    pub fn resolveOpt(self: Profile, comptime T: type, field: Field, client: ?T, fallback: ?T) ?T {
        if (self.forced(field)) return self.value(T, field);
        return client orelse self.value(T, field) orelse fallback;
    }

    pub fn resolve(self: Profile, comptime T: type, field: Field, client: ?T, fallback: T) T {
        return self.resolveOpt(T, field, client, fallback).?;
    }
};

/// Strict: an unknown field or key, or an out-of-range value, rejects the whole profile, so a
/// typo never drops one rule silently while the rest apply.
pub fn parseProfile(v: std.json.Value) !Profile {
    if (v != .object) return error.InvalidGenerationSettings;
    var p = Profile{};
    var it = v.object.iterator();
    while (it.next()) |e| {
        const field = std.meta.stringToEnum(Field, e.key_ptr.*) orelse return error.UnknownGenerationSetting;
        const rule = e.value_ptr.*;
        if (rule != .object) return error.InvalidGenerationSettings;
        for (rule.object.keys()) |k| {
            if (!std.mem.eql(u8, k, "value") and !std.mem.eql(u8, k, "ignore_client")) return error.InvalidGenerationSettings;
        }
        const force = rule.object.get("ignore_client") orelse std.json.Value{ .bool = false };
        if (force != .bool) return error.InvalidGenerationSettings;
        p.set(field, try parseValue(field, rule.object.get("value") orelse return error.InvalidGenerationSettings), force.bool);
    }
    return p;
}

fn parseValue(field: Field, raw: std.json.Value) !Value {
    switch (field) {
        .enable_thinking => return if (raw == .bool) .{ .boolean = raw.bool } else error.InvalidGenerationSettings,
        .reasoning_effort => {
            if (raw != .string) return error.InvalidGenerationSettings;
            return .{ .effort = std.meta.stringToEnum(Effort, raw.string) orelse return error.InvalidGenerationSettings };
        },
        else => {},
    }
    const n: f64 = switch (raw) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.InvalidGenerationSettings,
    };
    const lo: f64, const hi: f64 = switch (field) {
        .top_p, .min_p => .{ 0, 1 },
        .top_k => .{ 0, 1000 },
        .repeat_penalty => .{ 0.01, 10 },
        .max_tokens => .{ 0, std.math.maxInt(i32) },
        .reasoning_budget => .{ -1, std.math.maxInt(i32) },
        else => .{ 0, 2 },
    };
    const integral = field == .top_k or field == .max_tokens or field == .reasoning_budget;
    if (!std.math.isFinite(n) or n < lo or n > hi or (integral and @trunc(n) != n)) return error.InvalidGenerationSettings;
    return .{ .number = n };
}

test "generation settings: forced rule > client > rule > fallback; model rules replace global ones" {
    var global = Profile{};
    global.set(.temperature, .{ .number = 0.25 }, false);
    global.set(.max_tokens, .{ .number = 100 }, true);
    var model = Profile{};
    model.set(.max_tokens, .{ .number = 40 }, false);
    var p = Profile.overlay(global, model);
    try std.testing.expectEqual(@as(f32, 0.25), p.resolve(f32, .temperature, null, 1));
    try std.testing.expectEqual(@as(f32, 0), p.resolve(f32, .temperature, 0, 1)); // an explicit zero is a value
    try std.testing.expectEqual(@as(u32, 12), p.resolve(u32, .max_tokens, 12, 200)); // model unlocked the global lock
    try std.testing.expectEqual(@as(u32, 40), p.resolve(u32, .max_tokens, null, 200));
    try std.testing.expectEqual(@as(u32, 7), p.resolve(u32, .top_k, null, 7));
    p = Profile.overlay(global, .{});
    try std.testing.expectEqual(@as(u32, 100), p.resolve(u32, .max_tokens, 12, 200));
    try std.testing.expectEqual(@as(?f32, null), p.resolveOpt(f32, .min_p, null, null));
}

test "generation settings: a bad rule rejects the whole profile" {
    for ([_][]const u8{
        "{\"top_k\":{\"value\":-1}}",
        "{\"top_k\":{\"value\":2.5}}",
        "{\"enable_thinking\":{\"value\":false,\"ignore_client\":1}}",
        "{\"reasoning_budget\":{\"value\":-2}}",
        "{\"reasoning_effort\":{\"value\":\"banana\"}}",
        "{\"top_k\":{\"value\":0,\"ignore_clent\":true}}",
        "{\"temprature\":{\"value\":0.5}}",
    }) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, text, .{});
        defer parsed.deinit();
        try std.testing.expect(std.meta.isError(parseProfile(parsed.value)));
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"reasoning_budget":{"value":1024.0,"ignore_client":true},"reasoning_effort":{"value":"high"}}
    , .{});
    defer parsed.deinit();
    const p = try parseProfile(parsed.value);
    try std.testing.expectEqual(@as(i32, 1024), p.resolve(i32, .reasoning_budget, 4096, -1));
    try std.testing.expectEqual(Effort.high, p.value(Effort, .reasoning_effort).?);
}
