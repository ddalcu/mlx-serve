//! The `/v1/models` row fields clients used to derive for themselves: `agent_budget` (what an agent CLI declares,
//! `launch/budget.zig`), `modality` (which create pane a media pack opens in), `engines` (what can serve the pack:
//! the Model Settings engine picker's choices) and `plugin` (the outside plugin serving it, a badge). The same fields
//! on a ready row and an unloaded stub.

const std = @import("std");
const abi = @import("../../abi/engine.zig");
const agent_budget = @import("../launch/budget.zig");
const modality = @import("../registry/modality.zig");
const discovery = @import("../registry/discovery.zig");
const gen = @import("../media_jobs/gen.zig");
const served = @import("../engines/served.zig");
const ane_rt = @import("../../engines/ane/rt.zig");
const outside = @import("../plugins/loader.zig");

pub const Engine = @import("../registry/model_settings.zig").Engine;
pub const Engines = std.EnumSet(Engine);

/// `modality`: the capability spelling of the pack's media modality, with audio split by the route it serves
/// (speech, music or sound); null for a language model.
pub fn modalityName(io: std.Io, path: []const u8, model_type: []const u8) ?[]const u8 {
    const m = modality.fromType(model_type) orelse return null;
    if (m != .audio) return m.capability();
    if (gen.servesAudioRoute(io, path, model_type, .music)) return "music";
    if (gen.servesAudioRoute(io, path, model_type, .sound)) return "sound";
    return "speech";
}

/// Every engine whose probe accepts the pack at `path`. A GGUF goes to the embedded engines (and to MLX where
/// src/sdk/quant/gguf claims it, `mlx_gguf`); any other pack runs on MLX, and on the Neural Engine where this Mac
/// has one and the pack's layout runs there.
pub fn enginesFor(io: std.Io, allocator: std.mem.Allocator, path: []const u8, model_type: []const u8, mlx_gguf: bool) Engines {
    var set = Engines.empty;
    if (!std.fs.path.isAbsolute(path)) return set; // every probe opens the path absolute
    if (discovery.isGgufModelPath(io, path)) {
        if (mlx_gguf) set.insert(.mlx);
        const file = discovery.resolveGgufFile(io, allocator, path) catch return set;
        defer allocator.free(file);
        inline for (.{ .ds4, .llama }) |kind| if (served.table(kind)) |tbl| if (accepts(tbl.probe, file)) set.insert(kind);
        return set;
    }
    set.insert(.mlx);
    if (modality.fromType(model_type) == null and ane_rt.available() and
        accepts(served.table(.ane).?.probe, path)) set.insert(.ane);
    return set;
}

fn accepts(probe: anytype, path: []const u8) bool {
    var verdict: abi.Verdict = .{};
    return probe(abi.span(u8, path), abi.span(u8, ""), &verdict);
}

/// `,"agent_budget":{…},"modality":"…","engines":[…],"plugin":"…"`: the budget only on a row serving chat (`chat_ctx`
/// its advertised context, 0 when unknown), the modality only on a media row, the plugin only on an outside plugin's.
pub fn write(allocator: std.mem.Allocator, out: *std.ArrayList(u8), chat_ctx: ?u64, modality_name: ?[]const u8, engines: Engines, plugin: ?[]const u8) !void {
    if (chat_ctx) |ctx| {
        var buf: [256]u8 = undefined;
        try out.print(allocator, ",\"agent_budget\":{s}", .{agent_budget.json(&buf, ctx)});
    }
    if (modality_name) |m| try out.print(allocator, ",\"modality\":\"{s}\"", .{m});
    try out.appendSlice(allocator, ",\"engines\":[");
    var it = engines.iterator();
    var first = true;
    while (it.next()) |e| {
        if (!first) try out.append(allocator, ',');
        first = false;
        try out.print(allocator, "\"{s}\"", .{@tagName(e)});
    }
    try out.append(allocator, ']');
    // A plugin name is [a-z0-9_] (`abi.dylib.validName`): no escaping.
    if (plugin) |p| try out.print(allocator, ",\"plugin\":\"{s}\"", .{p});
}

/// The facts of one registry entry, its engines answered once (`LoadedModel.engines`). Registry mutex held.
pub fn forEntry(allocator: std.mem.Allocator, io: std.Io, entry: anytype, model_type: []const u8, chat_ctx: ?u64, mlx_gguf: bool) ![]u8 {
    if (entry.engines == null) entry.engines = enginesFor(io, allocator, entry.path, model_type, mlx_gguf).bits.mask;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try write(allocator, &out, chat_ctx, modalityName(io, entry.path, model_type), .{ .bits = .{ .mask = @intCast(entry.engines.?) } }, outside.servingName(io, entry.path));
    return out.toOwnedSlice(allocator);
}

// ── tests ──

const t = std.testing;

test "write: the budget on chat rows, the modality on media rows, the engines on every row, an outside plugin's badge" {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(t.allocator);
    var two = Engines.empty;
    two.insert(.ane);
    two.insert(.mlx);
    try write(t.allocator, &out, 24576, null, two, null);
    try t.expectEqualStrings(",\"agent_budget\":{\"context\":24576,\"output\":12288,\"compaction_reserve\":6144,\"too_small_for\":{\"claude\":65536,\"opencode\":32768,\"opencode2\":32768}},\"engines\":[\"mlx\",\"ane\"]", out.items);
    out.clearRetainingCapacity();
    try write(t.allocator, &out, null, "3d", Engines.initOne(.mlx), "toy_3d");
    try t.expectEqualStrings(",\"modality\":\"3d\",\"engines\":[\"mlx\"],\"plugin\":\"toy_3d\"", out.items);
    out.clearRetainingCapacity();
    try write(t.allocator, &out, 0, null, Engines.empty, null);
    try t.expectEqualStrings(",\"agent_budget\":{\"context\":32768,\"output\":8192,\"compaction_reserve\":8192,\"too_small_for\":{\"claude\":65536}},\"engines\":[]", out.items);
}

test "modalityName: the capability spelling, audio split by the route its backend serves" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const pack = pbuf[0..try tmp.dir.realPath(io, &pbuf)]; // the audio plugins' caps read no file
    try t.expectEqualStrings("image", modalityName(io, "", "flux2-klein-4b").?);
    try t.expectEqualStrings("3d", modalityName(io, "", "trellis2").?);
    try t.expectEqualStrings("decisions", modalityName(io, "", "kev").?);
    try t.expectEqualStrings("video", modalityName(io, "", "minimax_h3").?);
    try t.expectEqualStrings("speech", modalityName(io, pack, "kokoro").?);
    try t.expectEqualStrings("speech", modalityName(io, pack, "qwen3_tts").?);
    try t.expectEqualStrings("music", modalityName(io, pack, "acestep").?);
    try t.expectEqualStrings("music", modalityName(io, pack, "yue2").?);
    try t.expectEqualStrings("sound", modalityName(io, pack, "stable_audio3").?);
    try t.expect(modalityName(io, "", "qwen3_5") == null);
}

test "enginesFor: a safetensors pack runs on MLX; an unreadable GGUF path is claimed by no engine" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    // A media pack never probes the Neural Engine.
    try t.expect(enginesFor(io, t.allocator, dir, "flux2", false).eql(Engines.initOne(.mlx)));
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = "not a gguf" });
    const gguf = try std.fs.path.join(t.allocator, &.{ dir, "m.gguf" });
    defer t.allocator.free(gguf);
    try t.expect(enginesFor(io, t.allocator, gguf, "gguf", false).eql(Engines.empty));
    try t.expect(enginesFor(io, t.allocator, gguf, "gguf", true).eql(Engines.initOne(.mlx)));
}
