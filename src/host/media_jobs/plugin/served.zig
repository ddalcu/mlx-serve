//! The media plugins gen.zig serves, every registered one: the one place the host asks which media plugin a type or a
//! pack is.

const std = @import("std");
const abi = @import("../../../abi/media.zig");
const common = @import("../../../abi/common.zig");
const sdk_plugin = @import("../../../sdk/plugin.zig");
const registry = @import("../../../plugins/registry.zig");
const pack_dir = @import("../../registry/pack_dir.zig");
const FsPackDir = pack_dir.FsPackDir;
const log = @import("../../../sdk/util/log.zig");
const outside = @import("../../plugins/loader.zig");

pub const MediaTable = abi.MediaTable;

/// The registered media plugins, in registry order.
const entries = blk: {
    var out: []const sdk_plugin.Entry = &.{};
    for (registry.entries) |e| if (e.media != null) {
        out = out ++ .{e};
    };
    break :blk out;
};

/// Every media plugin's table.
pub const tables: [entries.len]*const MediaTable = blk: {
    var out: [entries.len]*const MediaTable = undefined;
    for (entries, &out) |e, *t| t.* = e.media.?;
    break :blk out;
};

/// Each table's model type: the first `model_type` its plugin claims (what discovery, the registry and the app call
/// the pack: `AudioVideo` for LTX, `minimax_music3` for Music 3), else its name. A pack claimed by its config.json
/// `model_type` reports that type instead (`claimType`).
pub const types: [entries.len][]const u8 = blk: {
    var out: [entries.len][]const u8 = undefined;
    for (entries, &out) |e, *ty| ty.* = if (e.plugin.claims.len > 0) e.plugin.claims[0] else e.plugin.name;
    break :blk out;
};

/// The served table for `model_type` (the type `claimType` reports for a pack), or null: a table's own type, else the
/// first served table that claims the type.
pub fn byName(model_type: []const u8) ?*const MediaTable {
    if (model_type.len > 0) for (outside.entries()) |e| if (e.media) |t| {
        if (std.mem.eql(u8, typeOf(t), model_type) or t.claims(.of(model_type), &pack_dir.none) == .by_type) return t;
    };
    for (tables, types) |t, ty| if (std.mem.eql(u8, ty, model_type)) return t;
    if (model_type.len == 0) return null;
    for (tables) |t| if (t.claims(.of(model_type), &pack_dir.none) == .by_type) return t;
    return null;
}

/// The model type a served table reports for its packs (an outside plugin's: its first declared type).
pub fn typeOf(t: *const MediaTable) []const u8 {
    for (tables, types) |x, ty| if (x == t) return ty;
    if (outside.ownerOf(t)) |p| if (p.manifest.claims.len > 0) return p.manifest.claims.slice()[0].slice();
    return t.name.slice();
}

/// `t`'s modality as the host enum `M` (same tag names).
pub fn modalityAs(comptime M: type, t: *const MediaTable) M {
    return std.meta.stringToEnum(M, @tagName(t.modality)).?;
}

/// The config.json bound discovery has always read under.
const max_config_bytes = 4 * 1024 * 1024;

/// The served table that wins `dir` under the registry's rank rules (every registered plugin claims, the highest rank
/// wins), or null: no claim, a tie, or a model plugin's win.
pub fn claim(dir: *const common.PackDir) ?*const MediaTable {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    return (claimWith(arena.allocator(), dir) orelse return null).t;
}

/// A claimed pack: its served table and the model type it reports (in `a`). A pack claimed by its config.json
/// `model_type` reports that type, as discovery always has; a pack claimed by its marker files reports the table's.
const Claimed = struct { t: *const MediaTable, model_type: []const u8 };

fn claimWith(a: std.mem.Allocator, dir: *const common.PackDir) ?Claimed {
    const cfg_len = dir.fileBytes("config.json");
    var why: common.Verdict = .{};
    const cfg: []const u8 = if (cfg_len == 0 or cfg_len > max_config_bytes) "" else blk: {
        const buf = a.alloc(u8, @intCast(cfg_len)) catch return null;
        break :blk dir.readFile("config.json", buf, &why) orelse "";
    };
    const mt = sdk_plugin.modelTypeOf(a, cfg) catch "";
    if (outside.winner(mt, cfg, dir)) |p| if (p.entry().media) |t| {
        var one: [1]common.Claim = undefined;
        return .{ .t = t, .model_type = if (sdk_plugin.claimsOf(&.{p.entry()}, mt, cfg, dir, &one)[0] == .by_type) mt else typeOf(t) };
    };
    var claims_buf: [registry.entries.len]common.Claim = undefined;
    const claims = sdk_plugin.claimsOf(&registry.entries, mt, cfg, dir, &claims_buf);
    const i = (common.pick(claims) catch {
        log.warn("[gen] two plugins claim the pack at one rank ({s}); it is not served as media\n", .{mt});
        return null;
    }) orelse return null;
    const t = registry.entries[i].media orelse return null;
    return .{ .t = t, .model_type = if (claims[i] == .by_type) mt else typeOf(t) };
}

/// The model type of the served pack in open folder `d` (owned by `allocator`), or null when no served table wins it.
pub fn claimTypeIn(io: std.Io, d: std.Io.Dir, allocator: std.mem.Allocator) ?[]u8 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const fs: FsPackDir = .{ .io = io, .root = d };
    const c = claimWith(arena.allocator(), &fs.packDir()) orelse return null;
    return allocator.dupe(u8, c.model_type) catch null;
}

/// `claim` over the folder at absolute `model_dir`.
pub fn claimPath(io: std.Io, model_dir: []const u8) ?*const MediaTable {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return null;
    defer d.close(io);
    return claimIn(io, d);
}

/// `claim` over an open folder.
pub fn claimIn(io: std.Io, d: std.Io.Dir) ?*const MediaTable {
    const fs: FsPackDir = .{ .io = io, .root = d };
    return claim(&fs.packDir());
}

// ── tests ──

const testing = std.testing;

test "served set: each served type resolves to its registered media table, nothing else does" {
    for (tables, types) |t, ty| {
        try testing.expectEqual(t, byName(ty).?);
        try testing.expectEqualStrings(ty, typeOf(t));
    }
    try testing.expectEqualStrings("AudioVideo", typeOf(byName("AudioVideo").?));
    try testing.expect(byName("gemma4") == null and byName("") == null and byName("ltx") == null);
    try testing.expectEqualStrings("flux2", byName("flux2").?.name.slice());
    try testing.expectEqualStrings("qwen_image", byName("qwen_image").?.name.slice());
    try testing.expectEqualStrings("hunyuan3d", byName("hunyuan3d_2_1").?.name.slice());
}

test "served set: a pack the served plugin claims routes to it; a stranger's pack and an empty dir do not" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try testing.expect(claimIn(io, tmp.dir) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"gemma4\"}" });
    try testing.expect(claimIn(io, tmp.dir) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"stable_audio3\"}" });
    try testing.expectEqualStrings("stable_audio3", claimIn(io, tmp.dir).?.name.slice());
}
