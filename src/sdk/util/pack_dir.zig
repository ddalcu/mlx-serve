//! The host's `PackDir` over a model directory on disk: existence, bounded reads, safetensors headers, folder
//! listings and file sizes, every path relative to the pack root (symlinks followed). No MLX.

const std = @import("std");
const common = @import("../../abi/common.zig");

const Str = common.Str;
const Verdict = common.Verdict;
const PackDir = common.PackDir;

/// A pack with no files: a plugin's claim on a bare `model_type`.
pub const none: PackDir = .{ .ctx = null, .exists = None.exists, .read = None.read, .safetensors_header = None.read, .list = None.read, .file_bytes = None.bytes };
const None = struct {
    fn exists(_: ?*anyopaque, _: Str) callconv(.c) bool {
        return false;
    }
    fn read(_: ?*anyopaque, _: Str, _: [*]u8, _: usize, _: *usize) callconv(.c) Verdict {
        return Verdict.refuse(.failed, "no files");
    }
    fn bytes(_: ?*anyopaque, _: Str) callconv(.c) u64 {
        return 0;
    }
};

pub const FsPackDir = struct {
    io: std.Io,
    root: std.Io.Dir,

    /// An absolute path only: a relative or empty one is refused (openDirAbsolute asserts on it). A GGUF picked by its
    /// file (`--mlx-gguf`) is the pack of its folder.
    pub fn open(io: std.Io, abs_path: []const u8) !FsPackDir {
        if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return error.NotAbsolute;
        const root = if (std.mem.endsWith(u8, abs_path, ".gguf")) std.fs.path.dirname(abs_path) orelse abs_path else abs_path;
        return .{ .io = io, .root = try std.Io.Dir.openDirAbsolute(io, root, .{}) };
    }

    pub fn close(self: *FsPackDir) void {
        self.root.close(self.io);
    }

    pub fn packDir(self: *const FsPackDir) PackDir {
        return .{ .ctx = @constCast(self), .exists = exists, .read = read, .safetensors_header = header, .list = list, .file_bytes = fileBytes };
    }

    fn of(ctx: ?*anyopaque) *const FsPackDir {
        return @ptrCast(@alignCast(ctx.?));
    }

    /// `rel` when it stays inside the pack: an absolute path would make `Dir` ignore the root.
    fn inside(rel: Str) ?[]const u8 {
        return if (std.fs.path.isAbsolute(rel.slice())) null else rel.slice();
    }

    fn exists(ctx: ?*anyopaque, rel: Str) callconv(.c) bool {
        const self = of(ctx);
        _ = self.root.statFile(self.io, inside(rel) orelse return false, .{}) catch return false;
        return true;
    }

    fn read(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const self = of(ctx);
        const size = fileBytes(ctx, rel);
        if (size == 0 and !exists(ctx, rel)) return Verdict.refuse(.failed, "no such file in the pack");
        if (size > cap) return Verdict.refuse(.too_large, "file larger than the read bound");
        const got = self.root.readFile(self.io, inside(rel) orelse return Verdict.refuse(.failed, "no such file in the pack"), buf[0..cap]) catch return Verdict.refuse(.failed, "the file does not read");
        len.* = got.len;
        return .ok;
    }

    fn header(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const self = of(ctx);
        var f = self.root.openFile(self.io, inside(rel) orelse return Verdict.refuse(.failed, "no such file in the pack"), .{}) catch return Verdict.refuse(.failed, "no such file in the pack");
        defer f.close(self.io);
        var rb: [4096]u8 = undefined;
        var rs = f.reader(self.io, &rb);
        const n = rs.interface.takeInt(u64, .little) catch return Verdict.refuse(.failed, "not a safetensors file");
        if (n > cap) return Verdict.refuse(.too_large, "safetensors header larger than the read bound");
        rs.interface.readSliceAll(buf[0..@intCast(n)]) catch return Verdict.refuse(.failed, "safetensors header past the end of the file");
        len.* = @intCast(n);
        return .ok;
    }

    fn list(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const self = of(ctx);
        const r = inside(rel) orelse return Verdict.refuse(.failed, "no such folder in the pack");
        var dir = self.root.openDir(self.io, if (r.len == 0) "." else r, .{ .iterate = true }) catch return Verdict.refuse(.failed, "no such folder in the pack");
        defer dir.close(self.io);
        var n: usize = 0;
        var it = dir.iterate();
        while (it.next(self.io) catch return Verdict.refuse(.failed, "the folder does not list")) |e| {
            const is_dir = switch (e.kind) {
                .directory => true,
                .file => false,
                .sym_link => blk: {
                    const st = dir.statFile(self.io, e.name, .{}) catch continue;
                    break :blk st.kind == .directory;
                },
                else => continue,
            };
            const need = e.name.len + @intFromBool(is_dir) + 1;
            if (n + need > cap) return Verdict.refuse(.too_large, "listing larger than the read bound");
            @memcpy(buf[n..][0..e.name.len], e.name);
            n += e.name.len;
            if (is_dir) {
                buf[n] = '/';
                n += 1;
            }
            buf[n] = 0;
            n += 1;
        }
        len.* = n;
        return .ok;
    }

    fn fileBytes(ctx: ?*anyopaque, rel: Str) callconv(.c) u64 {
        const self = of(ctx);
        const st = self.root.statFile(self.io, inside(rel) orelse return 0, .{}) catch return 0;
        return if (st.kind == .file) @intCast(st.size) else 0;
    }
};

// ── tests ──

const testing = std.testing;

test "pack dir: a folder on disk lists, sizes and reads like the in-memory pack" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "vae");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"a\":1}" });
    const st = "\x02\x00\x00\x00\x00\x00\x00\x00{}";
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "vae/w.safetensors", .data = st });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try tmp.dir.realPath(testing.io, &pbuf)];
    var fs = try FsPackDir.open(testing.io, path);
    defer fs.close();
    const dir = fs.packDir();
    try testing.expect(dir.has("config.json") and dir.has("vae") and !dir.has("nope"));
    try testing.expectEqual(@as(u64, st.len), dir.fileBytes("vae/w.safetensors"));
    try testing.expectEqual(@as(u64, 0), dir.fileBytes("vae"));
    var buf: [64]u8 = undefined;
    var why: Verdict = .{};
    try testing.expectEqualStrings("{\"a\":1}", dir.readFile("config.json", &buf, &why).?);
    try testing.expect(dir.readFile("config.json", buf[0..3], &why) == null and why.code == .too_large);
    var len: usize = 0;
    try testing.expect(dir.safetensors_header(dir.ctx, Str.of("vae/w.safetensors"), &buf, buf.len, &len).isOk());
    try testing.expectEqualStrings("{}", buf[0..len]);
    var names: [2][]const u8 = undefined;
    var it = dir.entries("", &buf).?;
    names[0] = it.next().?;
    names[1] = it.next().?;
    try testing.expect(it.next() == null);
    const sorted = std.mem.lessThan(u8, names[0], names[1]);
    try testing.expectEqualStrings("config.json", if (sorted) names[0] else names[1]);
    try testing.expectEqualStrings("vae/", if (sorted) names[1] else names[0]);
    var sub = dir.entries("vae", &buf).?;
    try testing.expectEqualStrings("w.safetensors", sub.next().?);
    try testing.expect(dir.entries("nope", &buf) == null);
    try testing.expectError(error.NotAbsolute, FsPackDir.open(testing.io, "rel"));
}

test "pack dir: an absolute path never reads outside the pack" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{}" });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try tmp.dir.realPath(testing.io, &pbuf)];
    var fs = try FsPackDir.open(testing.io, "/");
    defer fs.close();
    const dir = fs.packDir();
    var abs_buf: [std.fs.max_path_bytes]u8 = undefined;
    const abs = try std.fmt.bufPrint(&abs_buf, "{s}/config.json", .{path});
    var buf: [64]u8 = undefined;
    var why: Verdict = .{};
    try testing.expect(!dir.has(abs) and dir.fileBytes(abs) == 0);
    try testing.expect(dir.readFile(abs, &buf, &why) == null and why.code == .failed);
    var len: usize = 0;
    try testing.expect(!dir.safetensors_header(dir.ctx, Str.of(abs), &buf, buf.len, &len).isOk());
    try testing.expect(dir.entries(path, &buf) == null);
}
