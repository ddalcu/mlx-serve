//! A plugin's comptime declaration (`Plugin`), the checks the generated registry runs on each plugin root, and the
//! claim over registered tables. MLX-free: a plugin's `config.zig` imports it.

const std = @import("std");
const common = @import("../abi/common.zig");
const model_abi = @import("../abi/plugin.zig");
const media_abi = @import("../abi/media.zig");
const quant_abi = @import("../abi/quant.zig");

const Str = common.Str;
const Claim = common.Claim;
const PackDir = common.PackDir;

pub const Kind = enum { model, media, quant };

/// The declaration in a plugin's `plugin.zig` (`pub const plugin = sdk.Plugin{ ... }`). Beside it the root declares
/// `table` (the kind's table), `config` (its MLX-free `config.zig`, whose `claims` and, for a model, `parse` the
/// table points at) and `conformanceFixture` (the tiny pack the conformance suite runs).
pub const Plugin = struct {
    /// Equals the plugin's folder name.
    name: []const u8,
    kind: Kind,
    /// The config.json `model_type` values the plugin claims by type.
    claims: []const []const u8 = &.{},
    /// The checkpoints the gates load for this plugin.
    gate_models: []const []const u8 = &.{},
};

pub fn Table(comptime kind: Kind) type {
    return switch (kind) {
        .model => model_abi.ModelTable,
        .media => media_abi.MediaTable,
        .quant => quant_abi.QuantTable,
    };
}

fn tableVersion(comptime kind: Kind) u32 {
    return switch (kind) {
        .model => model_abi.version,
        .media => media_abi.version,
        .quant => quant_abi.version,
    };
}

fn oneOf(s: []const u8, set: []const []const u8) bool {
    for (set) |t| if (std.mem.eql(u8, s, t)) return true;
    return false;
}

/// A model `claims` by `model_type` alone.
pub fn claimsByType(comptime types: []const []const u8) *const fn (Str, Str, *const PackDir) callconv(.c) Claim {
    return &struct {
        fn f(model_type: Str, _: Str, _: *const PackDir) callconv(.c) Claim {
            return if (oneOf(model_type.slice(), types)) .by_type else .no;
        }
    }.f;
}

/// A media `claims` by `model_type` alone.
pub fn mediaClaimsByType(comptime types: []const []const u8) *const fn (Str, *const PackDir) callconv(.c) Claim {
    return &struct {
        fn f(model_type: Str, _: *const PackDir) callconv(.c) Claim {
            return if (oneOf(model_type.slice(), types)) .by_type else .no;
        }
    }.f;
}

fn fnPtr(comptime f: anytype) if (@typeInfo(@TypeOf(f)) == .@"fn") *const @TypeOf(f) else @TypeOf(f) {
    return if (@typeInfo(@TypeOf(f)) == .@"fn") &f else f;
}

/// Compile errors naming what plugin root `M` in folder `folder` gets wrong.
pub fn check(comptime M: type, comptime folder: []const u8) void {
    comptime {
        const w = "plugin " ++ folder;
        if (!@hasDecl(M, "plugin") or @TypeOf(M.plugin) != Plugin) @compileError(w ++ ": no `pub const plugin: sdk.Plugin`");
        const p = M.plugin;
        if (!std.mem.eql(u8, p.name, folder)) @compileError(w ++ ": declares the name " ++ p.name ++ ", not its folder's");
        if (!@hasDecl(M, "table") or @TypeOf(M.table) != Table(p.kind)) @compileError(w ++ ": no `pub const table: " ++ @typeName(Table(p.kind)) ++ "`");
        if (!std.mem.eql(u8, M.table.name.slice(), p.name)) @compileError(w ++ ": its table is named " ++ M.table.name.slice());
        if (M.table.size != @sizeOf(Table(p.kind)) or M.table.version != tableVersion(p.kind)) @compileError(w ++ ": table size or version is not this ABI's");
        if (!@hasDecl(M, "config")) @compileError(w ++ ": no `pub const config = @import(\"config.zig\")`");
        if (fnPtr(M.config.claims) != M.table.claims) @compileError(w ++ ": table.claims is not config.zig's `claims`");
        if (p.kind == .model and fnPtr(M.config.parse) != M.table.parse) @compileError(w ++ ": table.parse is not config.zig's `parse`");
        if (!@hasDecl(M, "conformanceFixture")) @compileError(w ++ ": no `conformanceFixture` for the conformance suite");
    }
}

/// One registered plugin as the host sees it: its declaration and the one table of its kind.
pub const Entry = struct {
    plugin: Plugin,
    model: ?*const model_abi.ModelTable = null,
    media: ?*const media_abi.MediaTable = null,
    quant: ?*const quant_abi.QuantTable = null,
};

/// The registry's entries for `mods` (plugin roots), checked against their folder `names`.
pub fn entries(comptime mods: []const type, comptime names: []const []const u8) [mods.len]Entry {
    var out: [mods.len]Entry = undefined;
    inline for (mods, names, 0..) |M, name, i| {
        check(M, name);
        out[i] = .{ .plugin = M.plugin };
        switch (M.plugin.kind) {
            .model => out[i].model = &M.table,
            .media => out[i].media = &M.table,
            .quant => out[i].quant = &M.table,
        }
    }
    return out;
}

/// `-Dunits`: each plugin's metadata (from its MLX-free `config.zig`) with its table taken by the symbol its own
/// compile unit exports, so the host imports no plugin code.
pub fn externEntries(comptime metas: []const Plugin) [metas.len]Entry {
    var out: [metas.len]Entry = undefined;
    inline for (metas, 0..) |p, i| {
        const sym = common.exportName(p.name, tableVersion(p.kind));
        out[i] = .{ .plugin = p };
        switch (p.kind) {
            .model => out[i].model = @extern(*const model_abi.ModelTable, .{ .name = sym }),
            .media => out[i].media = @extern(*const media_abi.MediaTable, .{ .name = sym }),
            .quant => out[i].quant = @extern(*const quant_abi.QuantTable, .{ .name = sym }),
        }
    }
    return out;
}

/// Exports each registered table as `mlx_serve_plugin_<name>_v<version>`.
pub fn exportAll(comptime mods: []const type) void {
    inline for (mods) |M| @export(&M.table, .{ .name = common.exportName(M.plugin.name, tableVersion(M.plugin.kind)) });
}

/// Each entry's claim on one pack, into `out` (one per entry). Quant plugins claim no packs.
pub fn claimsOf(list: []const Entry, model_type: []const u8, config_json: []const u8, dir: *const PackDir, out: []Claim) []Claim {
    for (list, out[0..list.len]) |e, *c| c.* = if (e.model) |t|
        t.claims(Str.of(model_type), Str.of(config_json), dir)
    else if (e.media) |t|
        t.claims(Str.of(model_type), dir)
    else
        .no;
    return out[0..list.len];
}

/// The root `model_type` of a config.json; empty when it has none or is not JSON.
pub fn modelTypeOf(a: std.mem.Allocator, config_json: []const u8) ![]const u8 {
    const C = struct { model_type: []const u8 = "" };
    const parsed = std.json.parseFromSliceLeaky(C, a, config_json, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return "",
    };
    return parsed.model_type;
}

// ── tests ──

const testing = std.testing;

test "sdk plugin: claims by type answers its own types only" {
    const f = claimsByType(&.{ "laguna", "laguna_text" });
    const dir: PackDir = undefined;
    try testing.expectEqual(Claim.by_type, f(Str.of("laguna_text"), .{}, &dir));
    try testing.expectEqual(Claim.no, f(Str.of("laguna2"), .{}, &dir));
    const g = mediaClaimsByType(&.{"flux2"});
    try testing.expectEqual(Claim.by_type, g(Str.of("flux2"), &dir));
    try testing.expectEqual(Claim.no, g(.{}, &dir));
}

test "sdk plugin: the root model_type of a config, empty for none or for a broken file" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqualStrings("laguna", try modelTypeOf(arena.allocator(), "{\"model_type\":\"laguna\",\"x\":[1]}"));
    try testing.expectEqualStrings("", try modelTypeOf(arena.allocator(), "{\"text_config\":{\"model_type\":\"q\"}}"));
    try testing.expectEqualStrings("", try modelTypeOf(arena.allocator(), "{"));
}
