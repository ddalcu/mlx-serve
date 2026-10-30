//! An outside plugin's dylib: the one symbol it exports (`Manifest`, under `symbol`) and the negotiation the host runs
//! before it reads anything else. Extern layouts, the C calling convention, `size` first, grows at its end.

const std = @import("std");
const common = @import("common.zig");
const model_abi = @import("plugin.zig");
const media_abi = @import("media.zig");
const quant_abi = @import("quant.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;

pub const version: u32 = 2;

/// The manifest's export name. Unversioned: a dylib built for another manifest version is refused by name, never
/// reported as "no plugin here".
pub const symbol = "mlx_serve_plugin";

pub const Kind = enum(u8) { model, media, quant };

pub const File = extern struct { name: Str, bytes: Str };

/// The plugin's conformance fixture (`sdk.fixture.Fixture` across the boundary), in memory the plugin owns until
/// `Manifest.fixture_free`.
pub const Fixture = extern struct {
    size: u32 = @sizeOf(Fixture),
    files: Span(File) = .{},
    body: Str = .{},
    block: Str = .{},
    ctx: ?*anyopaque = null,
};

pub const Manifest = extern struct {
    size: u32 = @sizeOf(Manifest),
    version: u32 = version,
    name: Str,
    kind: Kind,
    /// The MLX build the plugin was compiled against (the kit's `mlx.version`): one MLX per process.
    mlx: Str,
    /// The kind's table (`ModelTable`, `MediaTable` or `QuantTable`), `size` and `version` first.
    table: *const anyopaque,
    /// The `model_type`s it declares (`sdk.Plugin.claims`), for the conformance suite.
    claims: Span(Str) = .{},
    fixture: *const fn (out: *Fixture) callconv(.c) Verdict,
    fixture_free: *const fn (f: *Fixture) callconv(.c) void,
    /// The plugin's own version (its root's `version`), shown at install and in Settings; "" when it declares none.
    plugin_version: Str = .{},
};

/// The two fields every table opens with.
const Head = extern struct { size: u32, version: u32 };

/// Whether a struct the host reads by pointer is at least as large as the host's (a smaller one is an older build:
/// its missing tail would be read as garbage).
fn sized(comptime T: type, p: ?*const anyopaque) bool {
    const s = p orelse return true;
    return @as(*const u32, @ptrCast(@alignCast(s))).* >= @sizeOf(T);
}

/// The host's checks on a manifest, before any other member is read: refuses by name a manifest, table or hook table
/// of another version or a smaller size, another MLX, or a name a plugin cannot have.
pub fn negotiate(m: *const Manifest, host_mlx: []const u8) Verdict {
    if (m.version != version) return Verdict.refuse(.unsupported, "plugin manifest version is not this host's");
    if (m.size < @sizeOf(Manifest)) return Verdict.refuse(.unsupported, "plugin manifest is older than this host's");
    if (!validName(m.name.slice())) return Verdict.refuse(.failed, "plugin name is not [a-z0-9_], 1 to 64 bytes");
    if (!std.mem.eql(u8, std.mem.trim(u8, m.mlx.slice(), " \t\r\n"), std.mem.trim(u8, host_mlx, " \t\r\n")))
        return Verdict.refuse(.unsupported, "plugin was built against another MLX: rebuild it with this release's kit");
    const head: *const Head = @ptrCast(@alignCast(m.table));
    switch (m.kind) {
        .model => {
            if (head.version != model_abi.version) return Verdict.refuse(.unsupported, "model table version is not this host's");
            if (head.size < @sizeOf(model_abi.ModelTable)) return Verdict.refuse(.unsupported, "model table is older than this host's");
            const t: *const model_abi.ModelTable = @ptrCast(@alignCast(m.table));
            if (nullWord(t, "cache") or nullWord(t, "weights"))
                return Verdict.refuse(.failed, "model table has no cache or weight hook table: both are required, even empty");
            if (!sized(model_abi.CacheHooks, t.cache) or !sized(model_abi.WeightHooks, t.weights) or
                !sized(model_abi.SpecHooks, t.spec) or !sized(model_abi.MediaHooks, t.media) or
                !sized(model_abi.EmbedHooks, t.embed) or !sized(model_abi.DistHooks, t.dist) or
                !sized(model_abi.SpeechHooks, t.speech))
                return Verdict.refuse(.unsupported, "a model hook table is older than this host's");
            if (t.spec) |s| if (!sized(model_abi.HeadHooks, s.head) or !sized(model_abi.NativeDrafter, s.native_drafter))
                return Verdict.refuse(.unsupported, "a model hook table is older than this host's");
            if (!sized(model_abi.AuxHooks, t.cache.aux)) return Verdict.refuse(.unsupported, "model auxiliary hook table is older than this host's");
        },
        .media => {
            if (head.version != media_abi.version) return Verdict.refuse(.unsupported, "media table version is not this host's");
            if (head.size < @sizeOf(media_abi.MediaTable)) return Verdict.refuse(.unsupported, "media table is older than this host's");
        },
        .quant => {
            if (head.version != quant_abi.version) return Verdict.refuse(.unsupported, "quant table version is not this host's");
            if (head.size < @sizeOf(quant_abi.QuantTable)) return Verdict.refuse(.unsupported, "quant table is older than this host's");
        },
    }
    const t_name: *const Str = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(m.table)) + @sizeOf(Head)));
    if (!std.mem.eql(u8, t_name.slice(), m.name.slice())) return Verdict.refuse(.failed, "plugin table is named otherwise than its manifest");
    return .ok;
}

/// Whether a required pointer of `t` is null: a C plugin can leave one so, and Zig's non-optional field may not be
/// tested for null, so the word is read as bytes.
fn nullWord(t: *const model_abi.ModelTable, comptime field: []const u8) bool {
    const w: *const usize = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(t)) + @offsetOf(model_abi.ModelTable, field)));
    return w.* == 0;
}

/// A plugin name: `[a-z0-9_]`, 1 to 64 bytes (it names a file, a log line and a `/v1/models` badge).
pub fn validName(name: []const u8) bool {
    if (name.len == 0 or name.len > 64) return false;
    for (name) |c| if (!(std.ascii.isLower(c) or std.ascii.isDigit(c) or c == '_')) return false;
    return true;
}

// ── tests ──

const testing = std.testing;

fn noFixture(_: *Fixture) callconv(.c) Verdict {
    return .ok;
}
fn noFree(_: *Fixture) callconv(.c) void {}

fn quantTable(comptime name: []const u8) quant_abi.QuantTable {
    return .{ .name = .of(name), .claims = undefined, .validate = undefined, .bill = undefined, .open = undefined, .close = undefined, .load_tensor = undefined, .load_bank = undefined, .release = undefined, .matmul = undefined, .moe = undefined };
}

test "abi dylib: the manifest opens with size and version; the table name follows the head on every kind" {
    try testing.expectEqual(@as(usize, 0), @offsetOf(Manifest, "size"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(Manifest, "version"));
    inline for (.{ model_abi.ModelTable, media_abi.MediaTable, quant_abi.QuantTable }) |T| {
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
        try testing.expectEqual(@as(usize, 4), @offsetOf(T, "version"));
        try testing.expectEqual(@as(usize, @sizeOf(Head)), @offsetOf(T, "name"));
    }
}

test "abi dylib: negotiation accepts its own build and refuses each mismatch by name" {
    const pin = "mlx=64ea011cb65f mlxc=56b2d39fc831 target=26.2";
    var t = quantTable("fmt_x");
    var m: Manifest = .{ .name = .of("fmt_x"), .kind = .quant, .mlx = .of(pin ++ "\n"), .table = &t, .fixture = noFixture, .fixture_free = noFree };
    try testing.expect(negotiate(&m, pin).isOk());

    const Case = struct { []const u8, *const fn (*Manifest, *quant_abi.QuantTable) void };
    const cases = [_]Case{
        .{ "plugin manifest version is not this host's", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.version = version + 1;
            }
        }.f },
        .{ "plugin manifest is older than this host's", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.size -= 8;
            }
        }.f },
        .{ "plugin was built against another MLX: rebuild it with this release's kit", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.mlx = .of("mlx=000 mlxc=56b2d39fc831 target=26.2");
            }
        }.f },
        .{ "plugin name is not [a-z0-9_], 1 to 64 bytes", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.name = .of("../evil");
            }
        }.f },
        .{ "quant table version is not this host's", struct {
            fn f(_: *Manifest, q: *quant_abi.QuantTable) void {
                q.version += 1;
            }
        }.f },
        .{ "quant table is older than this host's", struct {
            fn f(_: *Manifest, q: *quant_abi.QuantTable) void {
                q.size -= 8;
            }
        }.f },
        .{ "plugin table is named otherwise than its manifest", struct {
            fn f(_: *Manifest, q: *quant_abi.QuantTable) void {
                q.name = .of("fmt_y");
            }
        }.f },
    };
    for (cases) |c| {
        var mm = m;
        var tt = t;
        mm.table = &tt;
        c[1](&mm, &tt);
        const v = negotiate(&mm, pin);
        try testing.expect(!v.isOk());
        try testing.expectEqualStrings(c[0], v.why.slice());
    }
    // A newer plugin's larger manifest is read up to the host's size.
    m.size += 16;
    try testing.expect(negotiate(&m, pin).isOk());
}

test "abi dylib: a model table with a hook table older than the host's is refused" {
    const pin = "p";
    var cache: model_abi.CacheHooks = .{};
    var weights: model_abi.WeightHooks = .{};
    var t: model_abi.ModelTable = .{ .name = .of("toy"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = &cache, .weights = &weights };
    const m: Manifest = .{ .name = .of("toy"), .kind = .model, .mlx = .of(pin), .table = &t, .fixture = noFixture, .fixture_free = noFree };
    try testing.expect(negotiate(&m, pin).isOk());
    cache.size -= 1;
    try testing.expectEqualStrings("a model hook table is older than this host's", negotiate(&m, pin).why.slice());
    cache.size += 1;
    const spec: model_abi.SpecHooks = .{ .size = 8 };
    t.spec = &spec;
    try testing.expect(!negotiate(&m, pin).isOk());
}

test "abi dylib: a model table without its cache or weight hook table is refused by name, never read" {
    const pin = "p";
    var cache: model_abi.CacheHooks = .{};
    var weights: model_abi.WeightHooks = .{};
    const t: model_abi.ModelTable = .{ .name = .of("toy"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = &cache, .weights = &weights };
    inline for (.{ "cache", "weights" }) |field| {
        var raw: [@sizeOf(model_abi.ModelTable)]u8 align(@alignOf(model_abi.ModelTable)) = undefined;
        @memcpy(&raw, std.mem.asBytes(&t));
        @memset(raw[@offsetOf(model_abi.ModelTable, field)..][0..@sizeOf(usize)], 0);
        const m: Manifest = .{ .name = .of("toy"), .kind = .model, .mlx = .of(pin), .table = &raw, .fixture = noFixture, .fixture_free = noFree };
        try testing.expectEqualStrings("model table has no cache or weight hook table: both are required, even empty", negotiate(&m, pin).why.slice());
    }
}

test "abi dylib: plugin names are lower-case words, never paths" {
    try testing.expect(validName("toy_2") and validName("_example"));
    const long: [65]u8 = @splat('x');
    for ([_][]const u8{ "", "Toy", "a/b", "a.b", "a-b", &long }) |n| try testing.expect(!validName(n));
}

test "abi dylib: auxiliary tables require the complete tail" {
    var aux: model_abi.AuxHooks = .{
        .enabled = undefined,
        .describe = undefined,
        .prefix_key = undefined,
        .begin_prompt = undefined,
        .import_plan = undefined,
        .export_arrays = undefined,
        .import_arrays = undefined,
        .reserve = undefined,
        .commit = undefined,
        .snapshot = undefined,
        .restore_at = undefined,
        .slice = undefined,
        .handoff = undefined,
        .free = undefined,
        .bytes = undefined,
    };
    var cache: model_abi.CacheHooks = .{ .aux = &aux };
    const weights: model_abi.WeightHooks = .{};
    const table: model_abi.ModelTable = .{ .name = .of("aux_fixture"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = &cache, .weights = &weights };
    const m: Manifest = .{ .name = .of("aux_fixture"), .kind = .model, .mlx = .of("p"), .table = &table, .fixture = noFixture, .fixture_free = noFree };
    try testing.expect(negotiate(&m, "p").isOk());
    for ([_]u32{
        @sizeOf(u32),
        @offsetOf(model_abi.AuxHooks, "enabled"),
        @offsetOf(model_abi.AuxHooks, "describe"),
        @offsetOf(model_abi.AuxHooks, "prefix_key"),
        @offsetOf(model_abi.AuxHooks, "begin_prompt"),
        @offsetOf(model_abi.AuxHooks, "import_plan"),
        @offsetOf(model_abi.AuxHooks, "export_arrays"),
        @offsetOf(model_abi.AuxHooks, "import_arrays"),
        @sizeOf(model_abi.AuxHooks) - 1,
    }) |size| {
        aux.size = size;
        const v = negotiate(&m, "p");
        try testing.expectEqual(Verdict.Code.unsupported, v.code);
        try testing.expectEqualStrings("model auxiliary hook table is older than this host's", v.why.slice());
    }
    aux.size = @sizeOf(model_abi.AuxHooks) + 8;
    try testing.expect(negotiate(&m, "p").isOk());
    var short: [@alignOf(model_abi.AuxHooks)]u8 align(@alignOf(model_abi.AuxHooks)) = @splat(0);
    @as(*u32, @ptrCast(&short)).* = @sizeOf(u32);
    cache.aux = @ptrCast(&short);
    try testing.expectEqual(Verdict.Code.unsupported, negotiate(&m, "p").code);
}
