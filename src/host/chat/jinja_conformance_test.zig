//! Jinja conformance: every `src/fixtures/*.jinja` template and the construct corpus render through jinja.cpp to the
//! same bytes as Python Jinja2 (transformers' chat-template environment). The references are Jinja2's renders,
//! checked in under `src/fixtures/jinja_conformance/`; `tests/jinja_conformance_ref.py` regenerates them.
//!
//! A template case is built exactly as the server builds it (`render.engineInput`, then `renderJinja`): its inputs
//! are stored in the reference and must still match, so a host change that alters what a template sees is caught
//! here too. Server-built cases must render (a raise is the silent generic fallback); "defaults" cases (only
//! bos/eos, the template's own defaults) and construct cases may raise only where Jinja2 raises.
//!
//!     zig build test -Dtest-filter="jinja conformance"

const std = @import("std");
const testing = std.testing;
const jinja_c = @import("jinja_c");
const data = @import("jinja_conformance");
const env = @import("../../env.zig");
const render = @import("render.zig");
const types = @import("types.zig");
const serializeMessagesJsonFor = @import("serialize.zig").serializeMessagesJsonFor;

const Message = types.Message;
const ToolCall = types.ToolCall;

/// Cases that differ from Jinja2 on purpose or until the engine learns the construct. A listed case that matches
/// fails too, so the list cannot go stale.
const known_differences = [_]struct { id: []const u8, reason: []const u8 }{
    .{ .id = "constructs/o03_tilde_none", .reason = "`~` drops None where Jinja2 writes \"None\"; no fixture concatenates a None" },
    .{ .id = "constructs/i06_slice_utf8", .reason = "strings are bytes: a slice cuts UTF-8 bytes, not code points" },
    .{ .id = "constructs/i09_len_utf8", .reason = "strings are bytes: `length` counts UTF-8 bytes, not code points" },
    .{ .id = "constructs/f02_default_none", .reason = "`default` replaces None too; the server sends no null context key, and keeping None turns renders into raises" },
    .{ .id = "constructs/f19_string_list_newline", .reason = "`string` of a list does not escape control characters as Python's repr does" },
    .{ .id = "constructs/f23_upper_unicode", .reason = "case mapping is ASCII only" },
    .{ .id = "constructs/f43_sort_case_default", .reason = "`sort` compares case-sensitively by default; Jinja2 folds case" },
    .{ .id = "constructs/t03_dict_iterable_sequence", .reason = "a dict is neither `iterable` nor `sequence`; every fixture tests `is mapping` first" },
    .{ .id = "constructs/p02_print_list", .reason = "`{{ list }}` prints the items joined, not Python's repr" },
};

const tools_json =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get the weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},"days":{"type":"integer"},"lat":{"type":"number"}},"required":["city"]}}}]
;

const call_paris = [_]ToolCall{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\": \"Paris\", \"days\": 3, \"lat\": 47.3769181}" }};
const calls_two = [_]ToolCall{
    .{ .id = "Call_A", .name = "get_weather", .arguments = "{\"city\": \"Paris\"}" },
    .{ .id = "call_a", .name = "get_weather", .arguments = "{\"city\": \"Rome\"}" },
};

const MessageSet = struct { name: []const u8, messages: []const Message, tools: bool };

const message_sets = [_]MessageSet{
    .{ .name = "simple", .tools = false, .messages = &.{
        .{ .role = "user", .content = "What is the capital of France?" },
    } },
    .{ .name = "multi", .tools = false, .messages = &.{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "Why is the sky blue?" },
        .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
        .{ .role = "user", .content = "And sunsets?" },
        .{ .role = "assistant", .content = "The light crosses more air." },
        .{ .role = "user", .content = "Thanks!" },
    } },
    .{ .name = "tool_round", .tools = true, .messages = &.{
        .{ .role = "system", .content = "You are a helpful assistant." },
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &call_paris },
        .{ .role = "tool", .content = "Sunny, 22C", .tool_call_id = "call_1" },
        .{ .role = "user", .content = "Thanks" },
    } },
    .{ .name = "parallel_tools", .tools = true, .messages = &.{
        .{ .role = "user", .content = "Weather in Paris and Rome?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls_two },
        .{ .role = "tool", .content = "Rome: cloudy, 19C", .tool_call_id = "call_a" },
        .{ .role = "tool", .content = "Paris: sunny, 22C", .tool_call_id = "Call_A" },
        .{ .role = "user", .content = "Which is warmer?" },
    } },
    .{ .name = "unicode", .tools = false, .messages = &.{
        .{ .role = "system", .content = "\u{3000}Réponds brièvement.\u{a0}" },
        .{ .role = "user", .content = "\u{3000}日本語で答えて 🙂\u{a0}" },
        .{ .role = "assistant", .content = "はい\u{2028}どうぞ\x00。" },
        .{ .role = "user", .content = "ok\u{85}" },
    } },
};

const Axis = struct { name: []const u8, thinking: bool, effort: ?[]const u8 = null, server: bool = true, sets: []const []const u8 = &.{} };

/// Empty `sets` = every message set.
const axes = [_]Axis{
    .{ .name = "think_off", .thinking = false },
    .{ .name = "think_on", .thinking = true },
    .{ .name = "defaults", .thinking = false, .server = false },
    .{ .name = "effort_none", .thinking = true, .effort = "none", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_minimal", .thinking = true, .effort = "minimal", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_low", .thinking = true, .effort = "low", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_medium", .thinking = true, .effort = "medium", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_high", .thinking = true, .effort = "high", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_xhigh", .thinking = true, .effort = "xhigh", .sets = &.{ "simple", "tool_round" } },
    .{ .name = "effort_max", .thinking = true, .effort = "max", .sets = &.{ "simple", "tool_round" } },
};

const bos = "<bos>";
const eos = "<eos>";
const defaults_extra = "{\"bos_token\":\"" ++ bos ++ "\",\"eos_token\":\"" ++ eos ++ "\"}";

fn takesSet(axis: Axis, set: []const u8) bool {
    if (axis.sets.len == 0) return true;
    for (axis.sets) |s| if (std.mem.eql(u8, s, set)) return true;
    return false;
}

const Inputs = struct { id: []const u8, in: render.EngineInput, messages: []const u8, tools: ?[]const u8, extra: []const u8, server: bool };

/// The template's cases, built through the host the way a request is.
fn buildInputs(a: std.mem.Allocator, config: *const types.ChatConfig) ![]Inputs {
    var out: std.ArrayList(Inputs) = .empty;
    for (message_sets) |set| for (axes) |axis| {
        if (!takesSet(axis, set.name)) continue;
        const tools: ?[]const u8 = if (set.tools) tools_json else null;
        const in = try render.engineInput(a, set.messages, config, tools, null, axis.thinking, axis.effort);
        try out.append(a, .{
            .id = try std.fmt.allocPrint(a, "{s}/{s}", .{ set.name, axis.name }),
            .in = in,
            .messages = try serializeMessagesJsonFor(a, in.messages, in.empty_content, config),
            .tools = in.tools_json,
            .extra = if (axis.server) in.extra_json else defaults_extra,
            .server = axis.server,
        });
    };
    return out.items;
}

fn sha256Hex(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

/// JSON text in one canonical spelling, so stored and built inputs compare by value.
fn canonical(a: std.mem.Allocator, json: ?[]const u8) ![]const u8 {
    const text = json orelse return "null";
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    return std.json.Stringify.valueAlloc(a, v, .{});
}

fn reference(name: []const u8) ?[]const u8 {
    for (data.references) |r| if (std.mem.eql(u8, r.name, name)) return r.bytes;
    return null;
}

fn known(id: []const u8) ?[]const u8 {
    for (known_differences) |k| if (std.mem.eql(u8, k.id, id)) return k.reason;
    return null;
}

const Tally = struct {
    cases: usize = 0,
    failures: usize = 0,
    matched_known: usize = 0,

    fn fail(self: *Tally, id: []const u8, comptime fmt: []const u8, args: anytype) void {
        self.failures += 1;
        std.debug.print("jinja conformance FAIL {s}: " ++ fmt ++ "\n", .{id} ++ args);
    }

    /// One case's verdict against the known-differences table.
    fn check(self: *Tally, id: []const u8, expected: ?[]const u8, got: ?[]const u8) void {
        self.cases += 1;
        const same = if (expected) |e| (got != null and std.mem.eql(u8, e, got.?)) else got == null;
        if (known(id)) |_| {
            if (same) {
                self.matched_known += 1;
                self.fail(id, "listed in known_differences but matches Jinja2 now: drop it from the list", .{});
            }
            return;
        }
        if (same) return;
        if (expected == null) return self.fail(id, "Jinja2 raises, jinja.cpp renders", .{});
        if (got == null) return self.fail(id, "jinja.cpp raises: {s}", .{if (jinja_c.jinja_last_error()) |e| std.mem.span(e) else "?"});
        const e = expected.?;
        const g = got.?;
        const at = std.mem.findDiff(u8, e, g) orelse 0;
        const lo = at -| 40;
        self.fail(id, "bytes differ at {d}\n  jinja2:    {f}\n  jinja.cpp: {f}", .{
            at, std.zig.fmtString(e[lo..@min(e.len, at + 40)]), std.zig.fmtString(g[lo..@min(g.len, at + 40)]),
        });
    }
};

fn dumpInputs(a: std.mem.Allocator, dir: []const u8, name: []const u8, tpl: []const u8, cases: []const Inputs) !void {
    var buf: std.ArrayList(u8) = .empty;
    try buf.print(a, "{{\"template\":\"{s}\",\"sha256\":\"{s}\",\"cases\":[", .{ name, try sha256Hex(a, tpl) });
    for (cases, 0..) |c, i| {
        if (i > 0) try buf.append(a, ',');
        try buf.print(a, "\n{{\"id\":\"{s}\",\"server\":{s},\"messages\":{s},\"tools\":{s},\"extra\":{s}}}", .{ c.id, if (c.server) "true" else "false", c.messages, c.tools orelse "null", c.extra });
    }
    try buf.appendSlice(a, "\n]}\n");
    const path = try std.fmt.allocPrint(a, "{s}/{s}.inputs.json", .{ dir, name });
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = path, .data = buf.items });
}

fn checkTemplate(a: std.mem.Allocator, tally: *Tally, name: []const u8, tpl: []const u8) !void {
    const config = types.ChatConfig{ .chat_template = tpl, .bos_token = bos, .eos_token = eos, .add_bos_token = false, .allocator = a };
    const cases = try buildInputs(a, &config);
    if (env.span(.MLX_SERVE_JINJA_CONFORMANCE_DUMP)) |dir| return dumpInputs(a, dir, name, tpl, cases);

    const ref_text = reference(name) orelse return tally.fail(name, "no reference: regenerate with tests/jinja_conformance_ref.py", .{});
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, ref_text, .{});
    if (!std.mem.eql(u8, ref.object.get("sha256").?.string, try sha256Hex(a, tpl)))
        return tally.fail(name, "the template changed since its reference: regenerate with tests/jinja_conformance_ref.py", .{});
    const stored = ref.object.get("cases").?.array.items;
    if (stored.len != cases.len) return tally.fail(name, "{d} cases built, {d} stored: regenerate the reference", .{ cases.len, stored.len });
    for (cases, stored) |c, s| {
        const id = try std.fmt.allocPrint(a, "{s}/{s}", .{ name, c.id });
        const so = s.object;
        if (!std.mem.eql(u8, so.get("id").?.string, c.id)) return tally.fail(id, "stored case is {s}: regenerate the reference", .{so.get("id").?.string});
        var stale: ?[]const u8 = null;
        inline for (.{ "messages", "tools", "extra" }) |field| {
            const built: ?[]const u8 = @field(c, field);
            const want = try std.json.Stringify.valueAlloc(a, so.get(field).?, .{});
            if (stale == null and !std.mem.eql(u8, try canonical(a, built), want)) stale = field;
        }
        if (stale) |field| {
            tally.fail(id, "the host now builds other {s} than the reference holds: regenerate it", .{field});
            continue;
        }
        const expected: ?[]const u8 = if (so.get("expected")) |e| e.string else null;
        if (c.server and expected == null) {
            tally.fail(id, "Jinja2 raises on a shape the server sends: {s}", .{so.get("raises").?.string});
            continue;
        }
        tally.check(id, expected, try render.renderJinja(a, c.in.messages, &config, c.in.tools_json, c.extra, c.in.empty_content));
    }
}

fn checkConstructs(a: std.mem.Allocator, tally: *Tally) !void {
    const ref_text = reference("constructs") orelse return tally.fail("constructs", "no reference: regenerate with tests/jinja_conformance_ref.py", .{});
    const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, ref_text, .{});
    for (ref.object.get("cases").?.array.items) |s| {
        const so = s.object;
        const id = try std.fmt.allocPrint(a, "constructs/{s}", .{so.get("id").?.string});
        const tpl = try a.dupeSentinel(u8, so.get("template").?.string, 0);
        const ctx = try a.dupeSentinel(u8, so.get("context").?.string, 0);
        var len: usize = 0;
        const ptr = jinja_c.jinja_render_chat(tpl.ptr, "[]", null, ctx.ptr, 0, &len);
        defer if (ptr != null) jinja_c.jinja_str_free(ptr);
        const got: ?[]const u8 = if (ptr == null) null else try a.dupe(u8, ptr[0..len]);
        tally.check(id, if (so.get("expected")) |e| e.string else null, got);
    }
}

test "jinja conformance: every fixture template and the construct corpus render like Python Jinja2" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tally: Tally = .{};
    for (data.templates) |t| try checkTemplate(a, &tally, t.name, t.bytes);
    if (env.span(.MLX_SERVE_JINJA_CONFORMANCE_DUMP) != null) return;
    try checkConstructs(a, &tally);
    for (known_differences) |k| if (k.reason.len == 0) tally.fail(k.id, "a known difference needs its reason", .{});
    if (tally.failures > 0) {
        std.debug.print("jinja conformance: {d} of {d} cases fail\n", .{ tally.failures, tally.cases });
        return error.JinjaConformance;
    }
}
