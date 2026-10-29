//! A pack's `config.json` through a plugin's own parse, for tests that run a module on a pack from disk.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const FsPackDir = @import("../util/pack_dir.zig").FsPackDir;

/// The `Config` plugin config module `C` (its `config.zig`) parses from the pack at `dir`.
pub fn packConfig(comptime C: type, io: std.Io, dir: []const u8) !C.Config {
    var fs = try FsPackDir.open(io, dir);
    defer fs.close();
    const pd = fs.packDir();
    const buf = try std.testing.allocator.alloc(u8, 1 << 20);
    defer std.testing.allocator.free(buf);
    var why: abi.Verdict = .{};
    const json = pd.readFile("config.json", buf, &why) orelse return error.ConfigUnreadable;
    var p: abi.Parsed = .{};
    if (!C.parse(&pd, .of(json), .{}, null, false, &p).isOk()) return error.ConfigRefused;
    defer C.freeParsed(&p);
    return C.of(&p).*;
}
