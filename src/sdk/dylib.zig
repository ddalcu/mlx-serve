//! What turns a plugin root into an outside plugin's dylib: `exportManifest` exports the root's table, declaration
//! and conformance fixture as the one `abi.dylib.Manifest` the host's loader reads. MLX-free.

const std = @import("std");
const common = @import("../abi/common.zig");
const dylib_abi = @import("../abi/dylib.zig");
const sdk_plugin = @import("plugin.zig");
const fixture = @import("testing/fixture.zig");

const Str = common.Str;
const Verdict = common.Verdict;

/// The manifest of plugin root `M` (the same root contract as a built-in: `plugin`, `table`, `config`,
/// `conformanceFixture`, and optionally `version`) built against MLX `mlx_pin`.
pub fn Manifest(comptime M: type, comptime mlx_pin: []const u8) type {
    comptime sdk_plugin.check(M, M.plugin.name);
    return struct {
        const claims = strs(M.plugin.claims);
        const models = strs(M.plugin.models);
        const options = optionsOf(M.plugin.options);

        pub const value: dylib_abi.Manifest = .{
            .name = Str.of(M.plugin.name),
            .kind = switch (M.plugin.kind) {
                .model => .model,
                .media => .media,
                .quant => .quant,
                .engine => .engine,
            },
            .mlx = Str.of(if (M.plugin.kind == .engine) dylib_abi.engine_pin else mlx_pin),
            .table = &M.table,
            .claims = .of(&claims),
            .fixture = &fixtureOf,
            .fixture_free = &fixtureFree,
            .plugin_version = Str.of(if (@hasDecl(M, "version")) M.version else ""),
            .models = .of(&models),
            .options = .of(&options),
        };

        const Owned = struct { arena: std.heap.ArenaAllocator };

        fn fixtureOf(out: *dylib_abi.Fixture) callconv(.c) Verdict {
            const owned = std.heap.c_allocator.create(Owned) catch return Verdict.refuse(.failed, "fixture: out of memory");
            owned.* = .{ .arena = .init(std.heap.c_allocator) };
            const a = owned.arena.allocator();
            const fx = M.conformanceFixture(a) catch |e| {
                owned.arena.deinit();
                std.heap.c_allocator.destroy(owned);
                return Verdict.named(.failed, @errorName(e));
            };
            const files = a.alloc(dylib_abi.File, fx.files.len) catch {
                owned.arena.deinit();
                std.heap.c_allocator.destroy(owned);
                return Verdict.refuse(.failed, "fixture: out of memory");
            };
            for (fx.files, files) |f, *o| o.* = .{ .name = .of(f.name), .bytes = .of(f.bytes) };
            out.* = .{ .files = .of(files), .body = .of(fx.body), .block = .of(fx.block), .ctx = owned };
            return .ok;
        }

        fn fixtureFree(f: *dylib_abi.Fixture) callconv(.c) void {
            const owned: *Owned = @ptrCast(@alignCast(f.ctx orelse return));
            owned.arena.deinit();
            std.heap.c_allocator.destroy(owned);
            f.* = .{};
        }
    };
}

fn strs(comptime list: []const []const u8) [list.len]Str {
    var out: [list.len]Str = undefined;
    for (list, &out) |c, *o| o.* = Str.of(c);
    return out;
}

fn optionsOf(comptime list: []const sdk_plugin.Option) [list.len]dylib_abi.Option {
    var out: [list.len]dylib_abi.Option = undefined;
    for (list, &out) |o, *x| x.* = .{ .name = Str.of(o.name), .kind = o.kind, .default_value = Str.of(o.default), .help = Str.of(o.help) };
    return out;
}

/// Exports `M`'s manifest as `abi.dylib.symbol`: the dylib's one entry point.
pub fn exportManifest(comptime M: type, comptime mlx_pin: []const u8) void {
    @export(&Manifest(M, mlx_pin).value, .{ .name = dylib_abi.symbol });
}

/// `f`'s files as the host's fixture type, borrowing the plugin's memory (valid until its `fixture_free`).
pub fn filesOf(a: std.mem.Allocator, f: *const dylib_abi.Fixture) !fixture.Fixture {
    const files = try a.alloc(fixture.File, f.files.len);
    for (f.files.slice(), files) |x, *o| o.* = .{ .name = x.name.slice(), .bytes = x.bytes.slice() };
    return .{ .files = files, .body = if (f.body.len > 0) f.body.slice() else "{}", .block = f.block.slice() };
}

// ── tests ──

const testing = std.testing;
const quant_abi = @import("../abi/quant.zig");
const engine_abi = @import("../abi/engine.zig");

const Toy = struct {
    pub const plugin = sdk_plugin.Plugin{ .name = "toy_fmt", .kind = .quant, .claims = &.{ "a", "b" } };
    pub const config = struct {
        pub fn claims(_: Str, _: Str) callconv(.c) common.Claim {
            return .no;
        }
    };
    pub const table = quant_abi.QuantTable{ .name = Str.of("toy_fmt"), .claims = &config.claims, .validate = undefined, .bill = undefined, .open = undefined, .close = undefined, .load_tensor = undefined, .load_bank = undefined, .release = undefined, .matmul = undefined, .moe = undefined };
    pub fn conformanceFixture(a: std.mem.Allocator) !fixture.Fixture {
        return .{ .files = try a.dupe(fixture.File, &.{.{ .name = "config.json", .bytes = try a.dupe(u8, "{}") }}), .block = "{\"mode\":\"toy\"}" };
    }
};

test "sdk dylib: the manifest carries the root's name, kind, table and claims, and negotiates with its own pin" {
    const m = &Manifest(Toy, "pin-1").value;
    try testing.expectEqualStrings("toy_fmt", m.name.slice());
    try testing.expectEqual(dylib_abi.Kind.quant, m.kind);
    try testing.expectEqual(@as(*const anyopaque, &Toy.table), m.table);
    try testing.expectEqualStrings("b", m.claims.slice()[1].slice());
    try testing.expect(dylib_abi.negotiate(m, "pin-1").isOk());
    try testing.expect(!dylib_abi.negotiate(m, "pin-2").isOk());
}

const ToyEngine = struct {
    pub const plugin = sdk_plugin.Plugin{ .name = "toy_engine", .kind = .engine, .models = &.{"Toy-1B"} };
    pub const table = engine_abi.Engine{ .probe = undefined, .load_bytes = undefined, .open = undefined, .close = undefined, .submit = undefined, .cancel = undefined, .step = null, .status = undefined, .memory = undefined, .report_json = null, .trim = null, .media_rows = null, .embed = null, .decide = null, .tokenize = null, .token_bytes = null, .chat_template = null, .render = null };
    pub fn conformanceFixture(a: std.mem.Allocator) !fixture.Fixture {
        return .{ .files = try a.dupe(fixture.File, &.{.{ .name = "config.json", .bytes = try a.dupe(u8, "{}") }}) };
    }
};

test "sdk dylib: an engine's manifest carries a pin no MLX matches, so a host that reads every pin refuses it" {
    const m = &Manifest(ToyEngine, "pin-1").value;
    try testing.expectEqualStrings(dylib_abi.engine_pin, m.mlx.slice());
    try testing.expect(dylib_abi.negotiate(m, "pin-1").isOk());
}

test "sdk dylib: the fixture crosses as plugin-owned files and is freed by the plugin" {
    const m = &Manifest(Toy, "pin-1").value;
    var f: dylib_abi.Fixture = .{};
    try testing.expect(m.fixture(&f).isOk());
    const fx = try filesOf(testing.allocator, &f);
    defer testing.allocator.free(fx.files);
    try testing.expectEqualStrings("{}", fx.file("config.json").?);
    try testing.expectEqualStrings("{\"mode\":\"toy\"}", fx.block);
    m.fixture_free(&f);
    try testing.expect(f.ctx == null and f.files.len == 0);
}
