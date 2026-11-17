//! The host's bank-fit key (`LoadCtx.bank_fit_key`, `sdk/quant/bank_fit_key.zig`): what a calibration measured on must
//! match this server for it to apply. A release build (`-Dgit-sha`) keys on its version, commit and MLX build, so a
//! fit shipped with a pack applies to every server of that release with the same chip, RAM and plugin builds; a dev
//! build keys on its executable.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const key = @import("../../../sdk/quant/bank_fit_key.zig");
const loader = @import("../../../host/plugins/loader.zig");
const outside = @import("../../../sdk/quant/outside.zig");
const offload = @import("../../../sdk/ane/offload.zig");
const common = @import("../../../abi/common.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;

/// `build`: `<version>+<sha> <MLX stamp> <mode> <kv>` for a release, `dev <engine build id> <mode> <kv>` otherwise.
pub fn buildPart(w: *std.Io.Writer, version: []const u8, git_sha: []const u8, stamp: []const u8, mode: std.builtin.OptimizeMode, kv_bits: u8, kv_group: u32, dev_id: []const u8) !void {
    if (git_sha.len > 0) {
        try w.print("{s}+{s} {s}", .{ version, git_sha, std.mem.trim(u8, stamp, " \t\r\n") });
    } else try w.print("dev {s}", .{dev_id});
    try w.print(" {s} ", .{@tagName(mode)});
    if (kv_bits == 0) try w.writeAll("kv16") else try w.print("kv{d}g{d}", .{ kv_bits, kv_group });
}

/// `machine`: the chip and the RAM in whole GiB.
pub fn machinePart(w: *std.Io.Writer, chip: []const u8, mem_bytes: u64) !void {
    try w.print("{s} {d}GB", .{ if (chip.len > 0) chip else "unknown", (mem_bytes + (1 << 29)) >> 30 });
}

/// A plugin part: `<name> <version> <dylib sha256>`.
pub fn pluginPart(w: *std.Io.Writer, p: loader.Plugin) !void {
    try w.print("{s} {s} {s}", .{ p.name(), p.version(), &p.sha256 });
}

/// The loaded outside plugin whose table is `table`, if any.
fn pluginOf(table: *const anyopaque) ?loader.Plugin {
    for (loader.loaded()) |p| if (p.manifest.table == table) return p;
    return null;
}

/// The key for `config`'s pack at `dir` (whose folder is `root`), in `a`.
pub fn hostKey(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, config: *const ModelConfig, dir: *const common.PackDir) ![]u8 {
    var parts: [4]std.Io.Writer.Allocating = .{ .init(a), .init(a), .init(a), .init(a) };
    defer for (&parts) |*p| p.deinit();
    const kv = @import("../../../host/http/server.zig").configuredKvQuantFor(config);
    const dev: []const u8 = if (build_options.git_sha.len == 0) @import("../spec/round_cost.zig").engineBuildId() else "";
    // The iOS graph stages no MLX stamp; it builds without `-Dgit-sha`, so it keys as a dev build.
    const stamp: []const u8 = if (comptime build_options.ios) "" else @import("mlx_stamp").bytes;
    try buildPart(&parts[0].writer, build_options.version, build_options.git_sha, stamp, builtin.mode, if (kv.scheme == .off) 0 else kv.bits, kv.group_size, dev);
    try machinePart(&parts[1].writer, offload.chipBrand(), offload.totalMemBytes());
    const config_json = try readConfig(a, dir);
    defer a.free(config_json);
    if (outside.claimingTable(a, config_json)) |t| {
        if (pluginOf(t)) |p| try pluginPart(&parts[2].writer, p) else try parts[2].writer.print("in-tree {s}", .{t.name.slice()});
    } else try parts[2].writer.writeAll("none");
    const b = config.plugin;
    if (b != null and b.?.outside) {
        if (pluginOf(b.?.table)) |p| try pluginPart(&parts[3].writer, p) else try parts[3].writer.writeAll("unknown");
    } else try parts[3].writer.writeAll("in-tree");
    const pack = try key.packDigest(a, io, root, dir);
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    try key.write(&out.writer, .{ .build = parts[0].written(), .machine = parts[1].written(), .quant_plugin = parts[2].written(), .model_plugin = parts[3].written(), .pack = &pack });
    return out.toOwnedSlice();
}

fn readConfig(a: std.mem.Allocator, dir: *const common.PackDir) ![]u8 {
    const n = dir.fileBytes("config.json");
    const buf = try a.alloc(u8, @intCast(n));
    errdefer a.free(buf);
    var why: common.Verdict = .{};
    _ = dir.readFile("config.json", buf, &why) orelse return error.BankFitKeyUnreadable;
    return buf;
}

const testing = std.testing;

test "fit key: a release build keys on version, commit, MLX stamp, mode and KV bits; a dev build on its build id" {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try buildPart(&w, "26.11.3", "e285ab55", "mlx=c69303b00f5f mlxc=56b2d39fc831 patch=613a4be4d66a target=26.2\n", .ReleaseFast, 0, 0, "");
    // The mode is its tag name (`fast` for ReleaseFast under Zig 0.17).
    try testing.expectEqualStrings("26.11.3+e285ab55 mlx=c69303b00f5f mlxc=56b2d39fc831 patch=613a4be4d66a target=26.2 " ++ @tagName(std.builtin.OptimizeMode.ReleaseFast) ++ " kv16", w.buffered());
    w = .fixed(&buf);
    try buildPart(&w, "26.11.3", "", "mlx=c69303b00f5f", .Debug, 8, 64, "0123456789abcdef");
    try testing.expectEqualStrings("dev 0123456789abcdef " ++ @tagName(std.builtin.OptimizeMode.Debug) ++ " kv8g64", w.buffered());
}

test "fit key: the machine is the chip and the RAM in whole GiB" {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try machinePart(&w, "Apple M5 Max", 137_438_953_472);
    try testing.expectEqualStrings("Apple M5 Max 128GB", w.buffered());
    w = .fixed(&buf);
    try machinePart(&w, "", 206_158_430_208);
    try testing.expectEqualStrings("unknown 192GB", w.buffered());
}
