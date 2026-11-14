//! An outside plugin's dylib: the one symbol it exports (`Manifest`, under `symbol`) and the negotiation the host runs
//! before it reads anything else. Extern layouts, the C calling convention, `size` first, grows at its end.

const std = @import("std");
const common = @import("common.zig");
const model_abi = @import("plugin.zig");
const media_abi = @import("media.zig");
const quant_abi = @import("quant.zig");
const engine_abi = @import("engine.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;

pub const version: u32 = 2;

/// The manifest's export name. Unversioned: a dylib built for another manifest version is refused by name, never
/// reported as "no plugin here".
pub const symbol = "mlx_serve_plugin";

/// `engine`: a whole engine of its own (`abi/engine.zig`), its own runtime on its own thread, for the models it names.
/// Open: a kind a later kit adds is refused by name (`negotiate`), never switched on.
pub const Kind = enum(u8) { model, media, quant, engine, _ };

/// An engine manifest's `mlx`: equal to no MLX build, so a host older than engine plugins (which checks every pin)
/// refuses one by name instead of switching on a kind it has no arm for.
pub const engine_pin = "engine plugin: links no MLX";

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
    /// The MLX build the plugin was compiled against (the kit's `mlx.version`): one MLX per process. An engine never
    /// calls MLX: its pin is not read.
    mlx: Str,
    /// The kind's table (`ModelTable`, `MediaTable`, `QuantTable` or `Engine`), `size` and `version` first.
    table: *const anyopaque,
    /// The `model_type`s it declares (`sdk.Plugin.claims`), for the conformance suite.
    claims: Span(Str) = .{},
    fixture: *const fn (out: *Fixture) callconv(.c) Verdict,
    fixture_free: *const fn (f: *Fixture) callconv(.c) void,
    /// The plugin's own version (its root's `version`), shown at install and in Settings; "" when it declares none.
    plugin_version: Str = .{},
    /// An engine's models: the model folders it serves, by name (`Qwen3-0.6B-4bit` serves `<models>/<org>/Qwen3-0.6B-4bit`),
    /// and no others. Read through `declaredModels`: a manifest built before it ends at `plugin_version`.
    models: Span(Str) = .{},
    /// The settings it reads from its environment (`getenv`, or the Host table's `lever`): the app draws them on the
    /// plugin's page, and the host sets each saved value before it opens the plugin, over one the shell set. Read
    /// through `declaredOptions`.
    options: Span(Option) = .{},

    /// The smallest manifest the host loads: one built before `models`, which declares no models and no options.
    pub const min_size = @offsetOf(Manifest, "models");

    pub fn declaredModels(m: *const Manifest) []const Str {
        return if (m.size >= @offsetOf(Manifest, "models") + @sizeOf(Span(Str))) m.models.slice() else &.{};
    }

    pub fn declaredOptions(m: *const Manifest) []const Option {
        return if (m.size >= @offsetOf(Manifest, "options") + @sizeOf(Span(Option))) m.options.slice() else &.{};
    }
};

pub const OptionKind = enum(u8) { bool, number, text, _ };

/// One setting (`Manifest.options`).
pub const Option = extern struct {
    /// The plugin's name upper-cased, `_`, then [A-Z0-9_] (`optionNameOk`), never `MLX_`, `MTL_`, `METAL_` or `DYLD_`. Two
    /// plugins whose names nest (`foo`, `foo_bar`) can still declare one name: the loader sets it for the first.
    name: Str,
    kind: OptionKind,
    /// The plugin's own value when none is set ("" when it names none), else a value of `kind` (`valueOk`).
    default_value: Str = .{},
    /// One line the app shows under the name.
    help: Str = .{},
};

/// Whether plugin `plugin` may declare option `name` (`Option.name`); at most 128 bytes.
pub fn optionNameOk(plugin: []const u8, name: []const u8) bool {
    if (name.len <= plugin.len + 1 or name.len > 128) return false;
    // mlx-serve's and MLX's own variables, Metal's, and the loader's (inherited by every child process).
    for ([_][]const u8{ "MLX_", "MTL_", "METAL_", "DYLD_" }) |reserved| if (std.mem.startsWith(u8, name, reserved)) return false;
    for (plugin, name[0..plugin.len]) |p, n| if (std.ascii.toUpper(p) != n) return false;
    if (name[plugin.len] != '_') return false;
    for (name[plugin.len + 1 ..]) |c| if (!(std.ascii.isUpper(c) or std.ascii.isDigit(c) or c == '_')) return false;
    return true;
}

/// Whether the host sets `value` for an option of `kind`: `0` or `1`, a plain decimal number, or text up to 1 KiB.
pub fn valueOk(kind: OptionKind, value: []const u8) bool {
    return switch (kind) {
        .bool => std.mem.eql(u8, value, "0") or std.mem.eql(u8, value, "1"),
        .number => plainNumber(value),
        .text => value.len <= 1024 and std.mem.indexOfScalar(u8, value, 0) == null,
        _ => false,
    };
}

/// `-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?`, what both the app and a plugin's parser read alike.
fn plainNumber(s: []const u8) bool {
    var i: usize = @intFromBool(s.len > 0 and s[0] == '-');
    i = digitsEnd(s, i) orelse return false;
    if (i < s.len and s[i] == '.') i = digitsEnd(s, i + 1) orelse return false;
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '-' or s[i] == '+')) i += 1;
        i = digitsEnd(s, i) orelse return false;
    }
    return i == s.len;
}

/// The index past the digits from `i`, or null when there is none.
fn digitsEnd(s: []const u8, i: usize) ?usize {
    var j = i;
    while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
    return if (j == i) null else j;
}

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
    if (m.version != version) return Verdict.refuse(.unsupported, "plugin manifest version is not this host's: rebuild it with this release's kit");
    if (m.size < Manifest.min_size) return Verdict.refuse(.unsupported, "plugin manifest is older than this host's: rebuild it with this release's kit");
    if (std.enums.tagName(Kind, m.kind) == null) return Verdict.refuse(.unsupported, "plugin kind is not one this host serves: a newer mlx-serve may");
    if (!validName(m.name.slice())) return Verdict.refuse(.failed, "plugin name is not [a-z0-9_], 1 to 64 bytes");
    for (m.declaredOptions()) |o| {
        if (!optionNameOk(m.name.slice(), o.name.slice()))
            return Verdict.refuse(.failed, "an option is not named after its plugin: `<PLUGIN NAME>_` then [A-Z0-9_], 128 bytes at most");
        if (std.enums.tagName(OptionKind, o.kind) == null) return Verdict.refuse(.failed, "an option's kind is not bool, number or text");
        if (o.default_value.len > 0 and !valueOk(o.kind, o.default_value.slice())) return Verdict.refuse(.failed, "an option's default is not a value of its kind");
    }
    if (m.kind != .engine and !std.mem.eql(u8, std.mem.trim(u8, m.mlx.slice(), " \t\r\n"), std.mem.trim(u8, host_mlx, " \t\r\n")))
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
            if (head.size < quant_abi.QuantTable.min_size) return Verdict.refuse(.unsupported, "quant table is older than this host's");
        },
        .engine => {
            if (head.version != engine_abi.version) return Verdict.refuse(.unsupported, "engine table version is not this host's");
            if (head.size < @sizeOf(engine_abi.Engine)) return Verdict.refuse(.unsupported, "engine table is older than this host's");
            if (m.declaredModels().len == 0) return Verdict.refuse(.failed, "an engine plugin names the models it serves (`models`)");
            // An engine's table carries no name: the manifest's is its name.
            return .ok;
        },
        _ => unreachable,
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
        .{ "plugin manifest version is not this host's: rebuild it with this release's kit", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.version = version + 1;
            }
        }.f },
        .{ "plugin manifest is older than this host's: rebuild it with this release's kit", struct {
            fn f(x: *Manifest, _: *quant_abi.QuantTable) void {
                x.size = Manifest.min_size - 8;
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
                q.size = quant_abi.QuantTable.min_size - 1;
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
    // A table built before `swiglu_limit_out` loads, the appended flag read as absent.
    var older = quantTable("fmt_x");
    older.swiglu_limit_out = true;
    older.size = quant_abi.QuantTable.min_size;
    var mo = m;
    mo.table = &older;
    try testing.expect(negotiate(&mo, pin).isOk());
    try testing.expect(!older.clampsOut());
    // A table built before the row-exact width: it loads, and declares none.
    var pre_rows = quantTable("fmt_x");
    pre_rows.exact_rows = 16;
    pre_rows.size = @offsetOf(quant_abi.QuantTable, "exact_rows");
    mo.table = &pre_rows;
    try testing.expect(negotiate(&mo, pin).isOk());
    try testing.expectEqual(@as(u32, 0), pre_rows.exactRows());
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

fn engineTable() engine_abi.Engine {
    return .{ .probe = undefined, .load_bytes = undefined, .open = undefined, .close = undefined, .submit = undefined, .cancel = undefined, .step = null, .status = undefined, .memory = undefined, .report_json = null, .trim = null, .media_rows = null, .embed = null, .decide = null, .tokenize = null, .token_bytes = null, .chat_template = null, .render = null };
}

test "abi dylib: an engine plugin needs no MLX pin and names the models it serves" {
    const e = engineTable();
    const models = [_]Str{.of("Qwen3-0.6B-4bit")};
    var m: Manifest = .{ .name = .of("raw_metal"), .kind = .engine, .mlx = .{}, .table = &e, .fixture = noFixture, .fixture_free = noFree, .models = .of(&models) };
    try testing.expect(negotiate(&m, "mlx=any").isOk());
    m.models = .{};
    const v = negotiate(&m, "mlx=any");
    try testing.expectEqual(Verdict.Code.failed, v.code);
    try testing.expectEqualStrings("an engine plugin names the models it serves (`models`)", v.why.slice());
    m.models = .of(&models);
    var old = e;
    old.version = engine_abi.version + 1;
    m.table = &old;
    try testing.expectEqualStrings("engine table version is not this host's", negotiate(&m, "mlx=any").why.slice());
    m.table = &e;
    m.kind = @enumFromInt(9);
    try testing.expectEqualStrings("plugin kind is not one this host serves: a newer mlx-serve may", negotiate(&m, "mlx=any").why.slice());
}

test "abi dylib: an option is named after its plugin and its default is a value of its kind" {
    const e = engineTable();
    const models = [_]Str{.of("Qwen3-0.6B-4bit")};
    var opts = [_]Option{
        .{ .name = .of("RAW_METAL_FRAC"), .kind = .number, .default_value = .of("0.25"), .help = .of("Share of the rows kept") },
        .{ .name = .of("RAW_METAL_FAST"), .kind = .bool },
    };
    var m: Manifest = .{ .name = .of("raw_metal"), .kind = .engine, .mlx = .{}, .table = &e, .fixture = noFixture, .fixture_free = noFree, .models = .of(&models), .options = .of(&opts) };
    try testing.expect(negotiate(&m, "mlx=any").isOk());
    opts[1].name = .of("RAWMETAL_FAST");
    try testing.expectEqualStrings("an option is not named after its plugin: `<PLUGIN NAME>_` then [A-Z0-9_], 128 bytes at most", negotiate(&m, "mlx=any").why.slice());
    opts[1].name = .of("RAW_METAL_FAST");
    opts[0].default_value = .of("a quarter");
    try testing.expectEqualStrings("an option's default is not a value of its kind", negotiate(&m, "mlx=any").why.slice());
    opts[0].default_value = .of("0.25");
    opts[0].kind = @enumFromInt(7);
    try testing.expectEqualStrings("an option's kind is not bool, number or text", negotiate(&m, "mlx=any").why.slice());
}

test "abi dylib: option names carry the plugin's name upper-cased; values follow their kind" {
    try testing.expect(optionNameOk("turboexl3_ds41", "TURBOEXL3_DS41_FRAC"));
    try testing.expect(optionNameOk("_example_engine", "_EXAMPLE_ENGINE_REPLY"));
    for ([_][]const u8{ "TURBOEXL3_DS41_", "TURBOEXL3_DS41", "TURBOEXL3_DS41_frac", "TURBOEXL3_DS4_FRAC", "OTHER_X", "TURBOEXL3_DS41_A-B" }) |n|
        try testing.expect(!optionNameOk("turboexl3_ds41", n));
    try testing.expect(!optionNameOk("mlx_serve", "MLX_SERVE_CTX") and !optionNameOk("dyld", "DYLD_INSERT_LIBRARIES"));
    try testing.expect(!optionNameOk("mlx", "MLX_METAL_FAST_SYNCH") and !optionNameOk("mtl", "MTL_DEBUG_LAYER") and !optionNameOk("metal", "METAL_DEVICE_WRAPPER_TYPE"));
    for ([_][]const u8{ "1", "-2.5", "1e3", "0.25E-2" }) |v| try testing.expect(valueOk(.number, v));
    for ([_][]const u8{ "", "x", "1.", ".5", "nan", "inf", "1_000", "0x10", "0.5\n" }) |v| try testing.expect(!valueOk(.number, v));
    try testing.expect(valueOk(.bool, "0") and valueOk(.bool, "1") and !valueOk(.bool, "true"));
    try testing.expect(valueOk(.text, "hi there") and !valueOk(.text, "a\x00b") and !valueOk(.text, &@as([1025]u8, @splat('x'))));
}

test "abi dylib: a manifest built before `models` loads, declaring no models and no options" {
    const t = quantTable("toy");
    var m: Manifest = .{ .name = .of("toy"), .kind = .quant, .mlx = .of("pin"), .table = &t, .fixture = noFixture, .fixture_free = noFree };
    m.size = Manifest.min_size;
    m.models.len = 7;
    m.options.len = 7;
    try testing.expect(negotiate(&m, "pin").isOk());
    try testing.expectEqual(@as(usize, 0), m.declaredModels().len);
    try testing.expectEqual(@as(usize, 0), m.declaredOptions().len);
    m.size = Manifest.min_size - 1;
    try testing.expectEqual(Verdict.Code.unsupported, negotiate(&m, "pin").code);
}
