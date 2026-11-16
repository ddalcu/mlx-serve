//! A bank calibration's key (`bank-fit.json` v3): the build, machine, plugins and pack its evidence ran on. The host
//! writes the key's canonical text (`LoadCtx.bank_fit_key`); a calibration applies only where every part is equal.
//! MLX-free.

const std = @import("std");
const common = @import("../../abi/common.zig");

pub const Parts = struct {
    build: []const u8,
    machine: []const u8,
    quant_plugin: []const u8,
    model_plugin: []const u8,
    pack: []const u8,
};

pub const names = [_][]const u8{ "build", "machine", "quant_plugin", "model_plugin", "pack" };

/// The canonical text: one `name=value` line per part, in `names` order; a newline in a value becomes a space.
pub fn write(w: *std.Io.Writer, p: Parts) !void {
    inline for (names) |n| {
        try w.writeAll(n ++ "=");
        for (@field(p, n)) |ch| try w.writeByte(if (ch == '\n' or ch == '\r') ' ' else ch);
        try w.writeByte('\n');
    }
}

/// The parts of canonical text `text`, or null when it is not one.
pub fn parse(text: []const u8) ?Parts {
    var vals: [names.len][]const u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '\n');
    inline for (names, 0..) |n, i| {
        const line = it.next() orelse return null;
        if (!std.mem.startsWith(u8, line, n ++ "=")) return null;
        vals[i] = line[n.len + 1 ..];
    }
    if ((it.next() orelse return null).len != 0 or it.next() != null) return null;
    return .{ .build = vals[0], .machine = vals[1], .quant_plugin = vals[2], .model_plugin = vals[3], .pack = vals[4] };
}

/// The key's id: the first 16 hex digits of its text's sha256.
pub fn id(text: []const u8) [16]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    return std.fmt.bytesToHex(d[0..8].*, .lower);
}

/// The pack file holding the calibration for key `text`: `bank-fit/<id>.json`.
pub fn fileName(buf: *[32]u8, text: []const u8) []const u8 {
    const i = id(text);
    return std.fmt.bufPrint(buf, "bank-fit/{s}.json", .{&i}) catch unreachable;
}

/// Bank manifests a pack may hold without naming them in `config.json`.
const manifest_files = [_][]const u8{ "expert-manifest-v2.json", "expert-manifest.json", "mtp-expert-manifest-v2.json", "mtp-expert-manifest.json" };

/// The pack's part: the sha256 of `config.json`, every bank manifest (`expert_streaming`'s `*manifest` fields and the
/// files above, each once, by name) and the bank's size (`experts.bin`), hex. `folder` is `dir`'s: a manifest (tens
/// of MB on a large pack) is hashed from it in pieces, so the heap holds only `config.json`.
pub fn packDigest(a: std.mem.Allocator, io: std.Io, folder: std.Io.Dir, dir: *const common.PackDir) ![64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const config = try readAll(a, dir, "config.json");
    defer a.free(config);
    mix(&h, "config.json", config);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var files: std.ArrayList([]const u8) = .empty;
    for (manifest_files) |f| if (dir.has(f)) try files.append(arena.allocator(), f);
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config, .{}) catch null;
    if (root) |r| if (r == .object) if (r.object.get("expert_streaming")) |es| if (es == .object) {
        var fi = es.object.iterator();
        while (fi.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, "manifest") and e.value_ptr.* == .string and dir.has(e.value_ptr.string))
            try files.append(arena.allocator(), e.value_ptr.string);
    };
    std.mem.sort([]const u8, files.items, {}, lessThan);
    var last: []const u8 = "";
    for (files.items) |f| {
        if (std.mem.eql(u8, f, last)) continue;
        last = f;
        try mixFile(&h, io, folder, f, dir.fileBytes(f));
    }
    var size: [20]u8 = undefined;
    mix(&h, "experts.bin", std.fmt.bufPrint(&size, "{d}", .{dir.fileBytes("experts.bin")}) catch unreachable);
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

fn lessThan(_: void, x: []const u8, y: []const u8) bool {
    return std.mem.lessThan(u8, x, y);
}

fn mix(h: *std.crypto.hash.sha2.Sha256, name: []const u8, bytes: []const u8) void {
    h.update(name);
    h.update(&[_]u8{0});
    h.update(std.mem.asBytes(&@as(u64, bytes.len)));
    h.update(bytes);
}

/// `mix` over the `len` bytes of `rel`, read in 64 KiB pieces.
fn mixFile(h: *std.crypto.hash.sha2.Sha256, io: std.Io, root: std.Io.Dir, rel: []const u8, len: u64) !void {
    h.update(rel);
    h.update(&[_]u8{0});
    h.update(std.mem.asBytes(&len));
    var file = root.openFile(io, rel, .{}) catch return error.BankFitKeyUnreadable;
    defer file.close(io);
    var rb: [4096]u8 = undefined;
    var rs = file.reader(io, &rb);
    var piece: [64 << 10]u8 = undefined;
    var got: u64 = 0;
    while (true) {
        const n = rs.interface.readSliceShort(&piece) catch return error.BankFitKeyUnreadable;
        h.update(piece[0..n]);
        got += n;
        if (n < piece.len) break;
    }
    if (got != len) return error.BankFitKeyUnreadable;
}

fn readAll(a: std.mem.Allocator, dir: *const common.PackDir, rel: []const u8) ![]u8 {
    const n = dir.fileBytes(rel);
    const buf = try a.alloc(u8, @intCast(n));
    errdefer a.free(buf);
    var why: common.Verdict = .{};
    const got = dir.readFile(rel, buf, &why) orelse return error.BankFitKeyUnreadable;
    if (got.len != buf.len) return error.BankFitKeyUnreadable;
    return buf;
}

const testing = std.testing;

test "bank fit key: the canonical text round-trips its parts, its id is the sha256's first 16 hex, its file is under bank-fit/" {
    const p: Parts = .{ .build = "26.11.3+e285ab55 mlx=c69303b00f5f ReleaseFast kv16", .machine = "Apple M5 Max 128GB", .quant_plugin = "turboexl3_ling 1.0.0 ab", .model_plugin = "in-tree", .pack = "cd" };
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try write(&w, p);
    const text = w.buffered();
    try testing.expectEqualStrings("build=26.11.3+e285ab55 mlx=c69303b00f5f ReleaseFast kv16\nmachine=Apple M5 Max 128GB\nquant_plugin=turboexl3_ling 1.0.0 ab\nmodel_plugin=in-tree\npack=cd\n", text);
    const back = parse(text).?;
    try testing.expectEqualStrings(p.machine, back.machine);
    try testing.expectEqualStrings(p.pack, back.pack);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(text, &d, .{});
    const want = std.fmt.bytesToHex(d[0..8].*, .lower);
    try testing.expectEqualStrings(&want, &id(text));
    var nb: [32]u8 = undefined;
    try testing.expectEqualStrings("bank-fit/" ++ want ++ ".json", fileName(&nb, text));
    // A newline in a value cannot add a line; a short or reordered text is not a key.
    w = .fixed(&buf);
    try write(&w, .{ .build = "a\nb", .machine = "m", .quant_plugin = "q", .model_plugin = "o", .pack = "p" });
    try testing.expectEqualStrings("a b", parse(w.buffered()).?.build);
    try testing.expect(parse("build=x\n") == null);
    try testing.expect(parse("machine=m\nbuild=b\nquant_plugin=q\nmodel_plugin=o\npack=p\n") == null);
}

test "bank fit key: the pack's digest reads config.json, its bank manifests and the bank's size, and moves when any changes" {
    const io = testing.io;
    const pack_dir = @import("../util/pack_dir.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"expert_streaming\":{\"manifest\":\"m.json\",\"mtp_manifest\":\"mtp.json\"}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m.json", .data = "{\"rows\":1}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.json", .data = "{\"rows\":2}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = "0123" });
    try tmp.dir.writeFile(io, .{ .sub_path = "README.md", .data = "a" });
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path);
    var fs = try pack_dir.FsPackDir.open(io, path[0..n]);
    defer fs.close();
    const dir = fs.packDir();
    const d0 = try packDigest(testing.allocator, io, fs.root, &dir);
    // A file outside the bank's set leaves it; the MTP manifest, the bank's size and an unnamed manifest move it.
    try tmp.dir.writeFile(io, .{ .sub_path = "README.md", .data = "b" });
    try testing.expectEqualStrings(&d0, &(try packDigest(testing.allocator, io, fs.root, &dir)));
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.json", .data = "{\"rows\":3}" });
    const d1 = try packDigest(testing.allocator, io, fs.root, &dir);
    try testing.expect(!std.mem.eql(u8, &d0, &d1));
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = "01234" });
    const d2 = try packDigest(testing.allocator, io, fs.root, &dir);
    try testing.expect(!std.mem.eql(u8, &d1, &d2));
    try tmp.dir.writeFile(io, .{ .sub_path = "expert-manifest-v2.json", .data = "{}" });
    try testing.expect(!std.mem.eql(u8, &d2, &(try packDigest(testing.allocator, io, fs.root, &dir))));
}

test "bank fit key: the pack's digest streams the bank's manifests, allocating none of them, and equals the whole-file digest" {
    const io = testing.io;
    const pack_dir = @import("../util/pack_dir.zig");
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const config = "{\"expert_streaming\":{\"manifest\":\"m.json\"}}";
    const manifest = try testing.allocator.alloc(u8, 256 << 10);
    defer testing.allocator.free(manifest);
    for (manifest, 0..) |*b, i| b.* = @truncate(i *% 131);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
    try tmp.dir.writeFile(io, .{ .sub_path = "m.json", .data = manifest });
    try tmp.dir.writeFile(io, .{ .sub_path = "experts.bin", .data = "0123" });
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path);
    var fs = try pack_dir.FsPackDir.open(io, path[0..n]);
    defer fs.close();
    const dir = fs.packDir();
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    mix(&h, "config.json", config);
    mix(&h, "m.json", manifest);
    mix(&h, "experts.bin", "4");
    const want = std.fmt.bytesToHex(h.finalResult(), .lower);
    // Bar: a heap far smaller than the manifest still digests the pack, to the whole-file digest's bytes.
    var heap: [16 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&heap);
    try testing.expectEqualStrings(&want, &(try packDigest(fba.allocator(), io, fs.root, &dir)));
}
