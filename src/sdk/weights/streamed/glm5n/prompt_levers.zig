//! glm5_next's prompt levers (the prefill audit's exact levers 1-5): construction-time routes set by the
//! pack's `expert_streaming.settings` block, read once, refused by name. The defaults are batch 27's measured bundle
//! (g22-pw at 190/217, lane on, draft_depth 3 + draft_gate 0.5: prefill 1,010 -> 1,073, +6.18 %, exact, ids identical):
//! `exl3_wave_rows` 2048, `prompt_route_cut` on, `prompt_static_order` sampled, `ttft_tail` deferred; g22-exl3's routes
//! (1024, off, off, off: the control) stay selectable by name.
//!   - `exl3_wave_rows` (the EXL3 family's DIG-X prompt waves): a wave's rows at most, default 2,048 (the control 1,024,
//!     MiMo's projection tile), 128 .. 4,096 in steps of the GEMMs' 128-row M tile. Exact: each M tile's rows accumulate their own K in the
//!     same order whatever wave holds them (a row's output does not depend on its wave). The bill's `expert_workspace`
//!     prices two waves in flight at the setting (bill.zig `workspaceBytes`); a value off the grid refuses the load.
//!   - `prompt_route_cut` (`off` | `on`): the wide call's rows bucketed by group once (the shared scheduler's
//!     `Wide.route_cut`: each group's bank passes over its own rows, the distinct experts and their counts in one pass)
//!     and, on the EXL3 family, each prompt call's rows ordered by slot with a stable counting sort (the block sort's
//!     order exactly). Host planning only: the same plans and waves, byte for byte.
//!   - `prompt_static_order` (`off` | `sampled`): P1's predictor (graph.zig `predictor`) routes at most `sampled_spans`
//!     evenly spaced spans of a prompt call (every span of a call of fewer): its counts only rank the read-ahead seed,
//!     never a row's experts. Token-identical.
//!   - `ttft_tail` (`off` | `deferred`): the prompt pass returns its logits before the MTP lane's catch-up and the phase
//!     change (the prompt scratch freed, every layer grown to its decode rows); both run at the request's first decode
//!     call, in the same order, on the same arrays, before its forward. Exact; the bill's phases are unchanged (the
//!     prompt phase's arrays live until the phase change either way).

const std = @import("std");

pub const RouteCut = enum { off, on };
pub const StaticOrder = enum { off, sampled };
pub const TtftTail = enum { off, deferred };

/// The DIG-X prompt wave's rows: the default (batch 27's measured bundle), the control's (g22-exl3's: MiMo's projection
/// tile; by name), the bounds and the grid (the GEMMs' M tile). 2,048 rows bill as 1,024 (bill.zig `tileBytes`: both
/// inside NVFP4's tile).
pub const default_wave_rows: u32 = 2048;
pub const control_wave_rows: u32 = 1024;
pub const min_wave_rows: u32 = 128;
pub const max_wave_rows: u32 = 4096;
pub const wave_rows_step: u32 = 128;
/// `prompt_static_order: sampled`: the spans of a prompt call P1's predictor routes at most.
pub const sampled_spans: u32 = 4;

/// A wave size the routes and the bill take: whole M tiles in [min_wave_rows, max_wave_rows].
pub fn validWaveRows(v: i64) bool {
    return v >= min_wave_rows and v <= max_wave_rows and @mod(v, wave_rows_step) == 0;
}

/// The keys this file reads (settings.zig `unknownSetting` admits them beside `setting_keys`).
pub const keys = [_][]const u8{ "exl3_wave_rows", "prompt_route_cut", "prompt_static_order", "ttft_tail" };

pub const Settings = struct {
    wave_rows: u32 = default_wave_rows,
    route_cut: RouteCut = .on,
    static_order: StaticOrder = .sampled,
    ttft_tail: TtftTail = .deferred,
    /// The settings named `exl3_wave_rows` (on the NVFP4 family the default is inert; a named size other than the
    /// control's refuses).
    wave_rows_named: bool = false,
    /// A key given a value it does not take (refused by name in `check`).
    malformed: struct { wave_rows: bool = false, route_cut: bool = false, static_order: bool = false, ttft_tail: bool = false } = .{},

    /// The settings object (after `unknownSetting`).
    pub fn apply(s: *Settings, o: std.json.ObjectMap) void {
        if (o.get("exl3_wave_rows")) |v| {
            s.wave_rows_named = true;
            if (v == .integer and validWaveRows(v.integer))
                s.wave_rows = @intCast(v.integer)
            else
                s.malformed.wave_rows = true;
        }
        if (o.get("prompt_route_cut")) |v| {
            if (routeOf(RouteCut, v)) |r| s.route_cut = r else s.malformed.route_cut = true;
        }
        if (o.get("prompt_static_order")) |v| {
            if (routeOf(StaticOrder, v)) |r| s.static_order = r else s.malformed.static_order = true;
        }
        if (o.get("ttft_tail")) |v| {
            if (routeOf(TtftTail, v)) |r| s.ttft_tail = r else s.malformed.ttft_tail = true;
        }
    }

    fn routeOf(comptime E: type, v: std.json.Value) ?E {
        return if (v == .string) std.meta.stringToEnum(E, v.string) else null;
    }

    /// Once, with the settings' other refusals: a malformed value, or a wave size other than the control's named on a family
    /// without DIG-X waves (there the default is inert).
    pub fn check(s: Settings, exl3: bool) !void {
        if (s.malformed.wave_rows) return error.Glm5NextExl3WaveRows;
        if (s.wave_rows_named and s.wave_rows != control_wave_rows and !exl3) return error.Glm5NextExl3WaveRowsFamily;
        if (s.malformed.route_cut) return error.Glm5NextPromptRouteCut;
        if (s.malformed.static_order) return error.Glm5NextPromptStaticOrder;
        if (s.malformed.ttft_tail) return error.Glm5NextTtftTail;
    }

    /// The construction's engagement line's fields (every switch named, defaults included).
    pub fn line(s: Settings, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "wave_rows={d} route_cut={s} static_order={s} ttft_tail={s}", .{ s.wave_rows, @tagName(s.route_cut), @tagName(s.static_order), @tagName(s.ttft_tail) }) catch buf[0..0];
    }
};

/// `prompt_static_order: sampled`: the indices of the spans P1's predictor routes out of a call's `n` spans: every span
/// when `n <= sampled_spans`, else `sampled_spans` evenly spaced ones (the middle of each of as many equal parts), in
/// ascending order. Returns how many it wrote to `out`.
pub fn sampledSpanIndices(n: usize, out: []usize) usize {
    if (n <= sampled_spans) {
        for (out[0..n], 0..) |*o, i| o.* = i;
        return n;
    }
    for (out[0..sampled_spans], 0..) |*o, i| o.* = (2 * i + 1) * n / (2 * sampled_spans);
    return sampled_spans;
}

const testing = std.testing;

fn parsed(text: []const u8) !Settings {
    const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, text, .{});
    defer v.deinit();
    var s: Settings = .{};
    s.apply(v.value.object);
    return s;
}

test "glm5_next prompt levers: the defaults are the measured bundle, the control by name; each key read, malformed values refused by name" {
    const d: Settings = .{};
    try d.check(false);
    try d.check(true);
    var buf: [128]u8 = undefined;
    try testing.expectEqualStrings("wave_rows=2048 route_cut=on static_order=sampled ttft_tail=deferred", d.line(&buf));
    try testing.expectEqualStrings("wave_rows=2048 route_cut=on static_order=sampled ttft_tail=deferred", (try parsed("{}")).line(&buf));
    // the control (g22-exl3's routes) by name, on either family
    const ctl = try parsed("{\"exl3_wave_rows\":1024,\"prompt_route_cut\":\"off\",\"prompt_static_order\":\"off\",\"ttft_tail\":\"off\"}");
    try ctl.check(true);
    try ctl.check(false);
    try testing.expectEqualStrings("wave_rows=1024 route_cut=off static_order=off ttft_tail=off", ctl.line(&buf));
    const all = try parsed("{\"exl3_wave_rows\":2048,\"prompt_route_cut\":\"on\",\"prompt_static_order\":\"sampled\",\"ttft_tail\":\"deferred\"}");
    try all.check(true);
    try testing.expectEqualStrings("wave_rows=2048 route_cut=on static_order=sampled ttft_tail=deferred", all.line(&buf));
    try testing.expectEqual(@as(u32, 4096), (try parsed("{\"exl3_wave_rows\":4096}")).wave_rows);
    try testing.expectEqual(@as(u32, 128), (try parsed("{\"exl3_wave_rows\":128}")).wave_rows);
    // Off the grid, out of range, or not an integer: refused by name.
    for ([_][]const u8{ "{\"exl3_wave_rows\":2000}", "{\"exl3_wave_rows\":0}", "{\"exl3_wave_rows\":8192}", "{\"exl3_wave_rows\":\"2048\"}", "{\"exl3_wave_rows\":2048.0}" }) |t|
        try testing.expectError(error.Glm5NextExl3WaveRows, (try parsed(t)).check(true));
    // A wave size other than the control's named on the NVFP4 family (no DIG-X waves) refuses; the default (unnamed) and
    // the control's do not.
    try testing.expectError(error.Glm5NextExl3WaveRowsFamily, (try parsed("{\"exl3_wave_rows\":2048}")).check(false));
    try (try parsed("{\"exl3_wave_rows\":1024}")).check(false);
    try testing.expectError(error.Glm5NextPromptRouteCut, (try parsed("{\"prompt_route_cut\":true}")).check(true));
    try testing.expectError(error.Glm5NextPromptStaticOrder, (try parsed("{\"prompt_static_order\":\"static\"}")).check(true));
    try testing.expectError(error.Glm5NextTtftTail, (try parsed("{\"ttft_tail\":\"on\"}")).check(true));
}

test "glm5_next prompt levers: the sampled predictor's spans (every span up to 4, then 4 evenly spaced)" {
    var out: [16]usize = undefined;
    for (1..5) |n| {
        try testing.expectEqual(n, sampledSpanIndices(n, &out));
        for (out[0..n], 0..) |o, i| try testing.expectEqual(i, o);
    }
    try testing.expectEqual(@as(usize, 4), sampledSpanIndices(16, &out));
    try testing.expectEqualSlices(usize, &.{ 2, 6, 10, 14 }, out[0..4]);
    // Distinct, ascending and in range at every call width up to the 16K call's 17 spans.
    for (5..18) |n| {
        try testing.expectEqual(@as(usize, 4), sampledSpanIndices(n, &out));
        for (out[0..4], 0..) |o, i| {
            try testing.expect(o < n);
            if (i > 0) try testing.expect(o > out[i - 1]);
        }
    }
}
