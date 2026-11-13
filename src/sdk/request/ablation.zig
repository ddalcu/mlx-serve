//! A request's directional ablation (`abi.Ablation`), resolved by the host from a profile the server loaded at launch.

const std = @import("std");

pub const Request = struct {
    /// The profile's unit direction, `[hidden]` f32; owned by the server's profile set, which outlives every request.
    direction: []const f32,
    /// In (0, `max_strength`]: a strength of 0 never reaches the engine (the request runs as a standard one).
    strength: f32,
    /// The profile's content hash with the strength folded in (`abi.Ablation.key`).
    key: u64,
    prefix: Prefix = .isolated,
};

/// How an ablated request shares the prefix cache with unablated ones. `isolated`: it restores and commits nothing
/// beside them (a module-owned cache keys its own entries by `key`). `shared`: it reads and writes the one cache as
/// if unablated, so a later request may restore state computed under the ablation.
pub const Prefix = enum { isolated, shared };

/// The strongest ablation a request or a model setting may ask for. Above 1 the projection removes more than the
/// component along the direction; the ceiling keeps a typo from clearing the whole stream.
pub const max_strength: f32 = 30;

/// An `ablation` object as a request body or a model setting spells it. `profile` borrows from the JSON; `prefix` is
/// a model setting's choice, never a request's.
pub const Field = struct { profile: []const u8, strength: f32 = 1, prefix: ?Prefix = null };

pub const Parsed = union(enum) { ok: Field, bad: []const u8 };

/// The keys an `ablation` object may name. A request takes `profile` and `strength`; a model setting also takes
/// `prefix_cache`, which only the operator chooses.
pub const Keys = struct { prefix_cache: bool = false };

/// `v` read as an `ablation` object, or the message naming what is wrong (static, or in `msg_buf`).
pub fn parseField(v: std.json.Value, msg_buf: []u8, keys: Keys) Parsed {
    if (v != .object) return .{ .bad = "ablation must be an object {\"profile\": string, \"strength\": number}" };
    const allowed: []const []const u8 = if (keys.prefix_cache) &.{ "profile", "strength", "prefix_cache" } else &.{ "profile", "strength" };
    var it = v.object.iterator();
    while (it.next()) |kv| {
        var known = false;
        for (allowed) |k| {
            if (std.mem.eql(u8, kv.key_ptr.*, k)) {
                known = true;
                break;
            }
        }
        if (!known) return .{ .bad = std.fmt.bufPrint(msg_buf, "unknown field 'ablation.{s}'", .{kv.key_ptr.*[0..@min(kv.key_ptr.len, 64)]}) catch "unknown field in ablation" };
    }
    const name_v = v.object.get("profile") orelse return .{ .bad = "ablation.profile is required" };
    if (name_v != .string) return .{ .bad = "ablation.profile must be a string" };
    const strength: f32 = if (v.object.get("strength")) |sv| switch (sv) {
        .float => |f| @floatCast(f),
        .integer => |i| if (i >= 0 and i <= max_strength) @floatFromInt(i) else -1,
        else => return .{ .bad = "ablation.strength must be a number in [0, 30]" },
    } else 1.0;
    if (!(strength >= 0 and strength <= max_strength)) return .{ .bad = "ablation.strength must be a number in [0, 30]" };
    var prefix: ?Prefix = null;
    if (keys.prefix_cache) {
        if (v.object.get("prefix_cache")) |pv| {
            if (pv != .string) return .{ .bad = "ablation.prefix_cache must be \"isolated\" or \"shared\"" };
            prefix = std.meta.stringToEnum(Prefix, pv.string) orelse
                return .{ .bad = "ablation.prefix_cache must be \"isolated\" or \"shared\"" };
        }
    }
    return .{ .ok = .{ .profile = name_v.string, .strength = strength, .prefix = prefix } };
}

// ── tests ──

test "ablation field: the three keys, their defaults, and each malformed value named" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [128]u8 = undefined;
    const request: Keys = .{};
    const setting: Keys = .{ .prefix_cache = true };
    const full = parseField(try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"profile\":\"p\",\"strength\":0.5,\"prefix_cache\":\"shared\"}", .{}), &buf, setting);
    try std.testing.expectEqualStrings("p", full.ok.profile);
    try std.testing.expectEqual(@as(f32, 0.5), full.ok.strength);
    try std.testing.expectEqual(@as(?Prefix, .shared), full.ok.prefix);
    const bare = parseField(try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"profile\":\"p\"}", .{}), &buf, setting);
    try std.testing.expect(bare.ok.strength == 1 and bare.ok.prefix == null);
    const top = parseField(try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"profile\":\"p\",\"strength\":30}", .{}), &buf, request);
    try std.testing.expectEqual(max_strength, top.ok.strength);
    const cases = [_]struct { text: []const u8, keys: Keys, says: []const u8 }{
        .{ .text = "7", .keys = request, .says = "must be an object" },
        .{ .text = "{\"profile\":\"p\",\"strenght\":1}", .keys = setting, .says = "unknown field 'ablation.strenght'" },
        .{ .text = "{\"strength\":1}", .keys = setting, .says = "profile is required" },
        .{ .text = "{\"profile\":1}", .keys = request, .says = "must be a string" },
        .{ .text = "{\"profile\":\"p\",\"strength\":31}", .keys = setting, .says = "[0, 30]" },
        .{ .text = "{\"profile\":\"p\",\"strength\":30.5}", .keys = request, .says = "[0, 30]" },
        .{ .text = "{\"profile\":\"p\",\"prefix_cache\":\"shared\"}", .keys = request, .says = "unknown field 'ablation.prefix_cache'" },
        .{ .text = "{\"profile\":\"p\",\"prefix_cache\":\"warm\"}", .keys = setting, .says = "prefix_cache must be" },
        .{ .text = "{\"profile\":\"p\",\"prefix_cache\":7}", .keys = setting, .says = "prefix_cache must be" },
    };
    for (cases) |c| {
        const got = parseField(try std.json.parseFromSliceLeaky(std.json.Value, a, c.text, .{}), &buf, c.keys);
        try std.testing.expect(std.mem.indexOf(u8, got.bad, c.says) != null);
    }
}
