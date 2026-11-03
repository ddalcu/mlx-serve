//! Stable Audio 3's claims, caps, bills and request fields: MLX-free (discovery and admission run them with no
//! device). Minimal table for the media driver's first adopter; the backend's own port extends it.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "stable_audio3",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"stabilityai/stable-audio-3-small"},
};

pub const model_types = [_][]const u8{"stable_audio3"};

pub const DEFAULT_STEPS: u32 = 8;
pub const MAX_STEPS: u32 = 50;
/// The small models' longest generation (the paper's limit for Small).
pub const MAX_SECONDS: f32 = 120;

/// By type when a config.json names it; by marker for the repo as Stability publishes it (no config.json,
/// stable-audio-tools' `model_config.json` with a T5Gemma conditioner).
pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    if (std.mem.eql(u8, model_type.slice(), model_types[0])) return .by_type;
    if (model_type.len == 0 and isStabilityRepo(dir)) return .by_marker;
    return .no;
}

fn isStabilityRepo(dir: *const abi.PackDir) bool {
    var buf: [1 << 20]u8 = undefined;
    var why: abi.Verdict = .{};
    const bytes = dir.readFile("model_config.json", &buf, &why) orelse return false;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch return false;
    if (v != .object) return false;
    const mt = v.object.get("model_type") orelse return false;
    if (mt != .string or !std.mem.eql(u8, mt.string, "diffusion_cond_inpaint")) return false;
    var cur = v;
    for ([_][]const u8{ "model", "conditioning", "configs" }) |k| {
        if (cur != .object) return false;
        cur = cur.object.get(k) orelse return false;
    }
    if (cur != .array) return false;
    for (cur.array.items) |c| {
        if (c != .object) continue;
        const t = c.object.get("type") orelse continue;
        if (t == .string and std.mem.eql(u8, t.string, "t5gemma")) return true;
    }
    return false;
}

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .sound = true } };
}

/// Everything stays resident and nothing converts: the safetensors at the root and one folder down.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

/// A request holds no more than the load's working set.
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
}

/// A job's fields: the prompt and the numbers, each bad one refused by name.
pub const Body = struct {
    prompt: []const u8,
    seconds: f32 = 10,
    steps: u32 = DEFAULT_STEPS,
    seed: u64 = 42,

    /// `a` owns the strings. The prompt is read the way the legacy endpoint read it (the first `"prompt"` string,
    /// unescaped by hand), the numbers as real JSON: a negative, fractional or quoted number is refused by name.
    pub fn parse(a: std.mem.Allocator, json: []const u8) union(enum) { ok: Body, bad: abi.Verdict } {
        const raw = rawString(json, "prompt") orelse return .{ .bad = bad("missing 'prompt' (a description of the sound)") };
        const prompt = unescape(a, raw) catch return .{ .bad = bad("invalid JSON body") };
        if (prompt.len == 0) return .{ .bad = bad("empty 'prompt'") };
        const v = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return .{ .bad = bad("invalid JSON body") };
        if (v != .object) return .{ .bad = bad("invalid JSON body") };
        const o = v.object;
        var b = Body{ .prompt = prompt };
        if (o.get("duration_seconds")) |x| {
            const d: f64 = switch (x) {
                .integer => |i| @floatFromInt(i),
                .float => |f| f,
                else => return .{ .bad = bad_seconds },
            };
            if (!(d > 0 and d <= MAX_SECONDS)) return .{ .bad = bad_seconds };
            b.seconds = @floatCast(d);
        }
        if (o.get("steps")) |x| {
            const n = wholeNumber(x) orelse return .{ .bad = bad_steps };
            if (n < 1 or n > MAX_STEPS) return .{ .bad = bad_steps };
            b.steps = @intCast(n);
        }
        if (o.get("seed")) |x| {
            const n = wholeNumber(x) orelse return .{ .bad = bad_seed };
            if (n < 0) return .{ .bad = bad_seed };
            b.seed = @intCast(n);
        }
        return .{ .ok = b };
    }

    fn bad(comptime why: []const u8) abi.Verdict {
        return abi.Verdict.refuse(.unsupported, why);
    }
    const bad_seconds = bad("'duration_seconds' must be a number in (0,120]");
    const bad_steps = bad("'steps' must be an integer in [1,50]");
    const bad_seed = bad("'seed' must be a non-negative integer");
};

/// The raw (still escaped) value of the first `"key": "..."` in `body`.
fn rawString(body: []const u8, key: []const u8) ?[]const u8 {
    var pat_buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&pat_buf, "\"{s}\"", .{key}) catch return null;
    var i = (std.mem.indexOf(u8, body, pat) orelse return null) + pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < body.len) : (i += 1) {
        if (body[i] == '\\') {
            i += 1;
            continue;
        }
        if (body[i] == '"') return body[start..i];
    }
    return null;
}

/// JSON string escapes, one `\uXXXX` at a time (the legacy endpoint's reading).
fn unescape(a: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        if (raw[i] != '\\') {
            try out.append(a, raw[i]);
            continue;
        }
        i += 1;
        if (i >= raw.len) break;
        switch (raw[i]) {
            'n' => try out.append(a, '\n'),
            't' => try out.append(a, '\t'),
            'r' => try out.append(a, '\r'),
            'u' => if (i + 4 < raw.len) {
                const cp = std.fmt.parseInt(u21, raw[i + 1 .. i + 5], 16) catch 0;
                var bb: [4]u8 = undefined;
                try out.appendSlice(a, bb[0 .. std.unicode.utf8Encode(cp, &bb) catch 0]);
                i += 4;
            },
            else => try out.append(a, raw[i]),
        }
    }
    return out.items;
}

/// An integer, or a float with no fractional part; null for anything else.
fn wholeNumber(v: std.json.Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (@trunc(f) == f and @abs(f) < 9e18) @intFromFloat(f) else null,
        else => null,
    };
}

// ── tests ──

const testing = std.testing;

test "stable_audio3 body: defaults, real JSON numbers, and every bad field refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const d = Body.parse(a, "{\"prompt\":\"rain\"}").ok;
    try testing.expectEqualStrings("rain", d.prompt);
    try testing.expect(d.seconds == 10 and d.steps == DEFAULT_STEPS and d.seed == 42);
    const n = Body.parse(a, "{\"prompt\":\"r\",\"duration_seconds\":1.5e1,\"steps\":8.0,\"seed\":7}").ok;
    try testing.expect(n.seconds == 15 and n.steps == 8 and n.seed == 7);
    const cases = [_][2][]const u8{
        .{ "{}", "missing 'prompt'" },                               .{ "not json", "missing 'prompt'" },
        .{ "{\"prompt\":", "missing 'prompt'" },                     .{ "{\"prompt\":\"r\",}", "JSON" },                                 .{ "{\"prompt\":\"\"}", "empty 'prompt'" },
        .{ "{\"prompt\":\"r\",\"duration_seconds\":0}", "duration_seconds" }, .{ "{\"prompt\":\"r\",\"duration_seconds\":\"10\"}", "duration_seconds" },
        .{ "{\"prompt\":\"r\",\"steps\":51}", "steps" },               .{ "{\"prompt\":\"r\",\"steps\":4.9}", "steps" },
        .{ "{\"prompt\":\"r\",\"seed\":-1}", "seed" },
    };
    for (cases) |c| {
        const v = Body.parse(a, c[0]).bad;
        try testing.expect(v.code == .unsupported and std.mem.indexOf(u8, v.why.slice(), c[1]) != null);
    }
}
