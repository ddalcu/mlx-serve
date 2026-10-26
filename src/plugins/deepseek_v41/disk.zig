//! Finished conversations on the SSD: a conversation the plugin evicts from RAM (and every kept one at unload) is
//! written here, and a prompt nothing in RAM resumes is matched against these and read back (`plugin.resumePrompt`).
//! One safetensors file per conversation (`deepseek_v41.exportKept`) and a sidecar with its prompt ids and image key,
//! which the index reads at load. The directory is per build and pack (`fingerprint`), so a rebuilt server never reads
//! a state an older build wrote; other builds' directories are removed at load. `--prefix-cache-disk` caps the bytes,
//! least recently used first.

const std = @import("std");
const dsv41 = @import("deepseek_v41.zig");
const mlx = @import("../../mlx/ffi.zig");

const gpa = std.heap.c_allocator;
const Io = std.Io;

pub const Entry = struct {
    id: u64,
    ids: []u32,
    vision_key: u64,
    bytes: u64,
    tick: u64,
};

pub const Disk = struct {
    root: []u8,
    cap: u64,
    entries: std.ArrayList(Entry) = .empty,
    next_id: u64 = 1,
    tick: u64 = 0,

    pub fn deinit(d: *Disk) void {
        for (d.entries.items) |e| gpa.free(e.ids);
        d.entries.deinit(gpa);
        gpa.free(d.root);
    }

    fn path(d: *const Disk, buf: []u8, id: u64, ext: []const u8) ![:0]u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{d}.{s}", .{ d.root, id, ext }, 0);
    }

    fn bytesHeld(d: *const Disk) u64 {
        var n: u64 = 0;
        for (d.entries.items) |e| n += e.bytes;
        return n;
    }

    fn remove(d: *Disk, io: Io, i: usize) void {
        const e = d.entries.orderedRemove(i);
        defer gpa.free(e.ids);
        var b: [512]u8 = undefined;
        for ([_][]const u8{ "safetensors", "ids" }) |ext| {
            const p = d.path(&b, e.id, ext) catch continue;
            Io.Dir.cwd().deleteFile(io, p) catch {};
        }
    }

    /// Writes a conversation (state `st`, snapshot `sn`), then drops the least recently used until the cap holds.
    pub fn put(d: *Disk, io: Io, m: *dsv41.Dsv41Model, st: *const dsv41.State, sn: *const dsv41.Snap) !void {
        const id = d.next_id;
        d.next_id += 1;
        var b: [512]u8 = undefined;
        try dsv41.exportKept(m, st, sn, try d.path(&b, id, "safetensors"));
        const bytes = (Io.Dir.cwd().statFile(io, try d.path(&b, id, "safetensors"), .{}) catch return error.Dsv41KeptWrite).size;
        try writeSidecar(io, try d.path(&b, id, "ids"), sn.vision_key, sn.ids);
        d.tick += 1;
        try d.entries.append(gpa, .{ .id = id, .ids = try gpa.dupe(u32, sn.ids), .vision_key = sn.vision_key, .bytes = bytes, .tick = d.tick });
        while (d.entries.items.len > 1 and d.bytesHeld() > d.cap) {
            var lru: usize = 0;
            for (d.entries.items, 0..) |e, i| if (e.tick < d.entries.items[lru].tick) {
                lru = i;
            };
            d.remove(io, lru);
        }
    }

    /// The entry whose prompt is the longest proper prefix of `prompt` with the same images, or null.
    pub fn best(d: *const Disk, m: *dsv41.Dsv41Model, prompt: []const u32, rows: ?mlx.mlx_array) !?usize {
        var pick: ?usize = null;
        var pick_len: usize = 0;
        for (d.entries.items, 0..) |e, i| {
            if (e.ids.len <= pick_len or e.ids.len >= prompt.len) continue;
            if (!std.mem.eql(u32, e.ids, prompt[0..e.ids.len])) continue;
            if (e.vision_key != try dsv41.visionKey(rows, m.imageRows(e.ids))) continue;
            pick = i;
            pick_len = e.ids.len;
        }
        return pick;
    }

    /// Reads entry `i` back as a state and its snapshot of identity `gen`, and removes it from the disk: the caller
    /// keeps it in RAM, and evicting it later writes it again.
    pub fn take(d: *Disk, io: Io, m: *dsv41.Dsv41Model, i: usize, gen: u64) !dsv41.Imported {
        var b: [512]u8 = undefined;
        const got = dsv41.importKept(m, try d.path(&b, d.entries.items[i].id, "safetensors"), gen);
        d.remove(io, i);
        return got;
    }
};

fn writeSidecar(io: Io, p: [:0]const u8, vision_key: u64, ids: []const u32) !void {
    var f = try Io.Dir.cwd().createFile(io, p, .{});
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var w = f.writer(io, &buf);
    try w.interface.writeAll(dsv41.kept_format);
    try w.interface.writeInt(u64, vision_key, .little);
    try w.interface.writeInt(u64, ids.len, .little);
    try w.interface.writeSliceEndian(u32, ids, .little);
    try w.interface.flush();
}

fn readSidecar(io: Io, p: []const u8) !struct { vision_key: u64, ids: []u32 } {
    const data = try Io.Dir.cwd().readFileAlloc(io, p, gpa, .limited(64 << 20));
    defer gpa.free(data);
    const head = dsv41.kept_format.len + 16;
    if (data.len < head or !std.mem.eql(u8, data[0..dsv41.kept_format.len], dsv41.kept_format)) return error.Dsv41KeptFormat;
    const vk = std.mem.readInt(u64, data[dsv41.kept_format.len..][0..8], .little);
    const n = std.mem.readInt(u64, data[dsv41.kept_format.len + 8 ..][0..8], .little);
    if (data.len != head + n * 4) return error.Dsv41KeptFormat;
    const ids = try gpa.alloc(u32, n);
    @memcpy(std.mem.sliceAsBytes(ids), data[head..]);
    return .{ .vision_key = vk, .ids = ids };
}

/// What names a directory: the format, this build and the pack (its directory and config bytes). A state written by
/// another build or pack is never read.
pub fn fingerprint(io: Io, model_dir: []const u8, build: []const u8) !u64 {
    var h = std.hash.Wyhash.init(0xd5_41_ce);
    h.update(dsv41.kept_format);
    h.update(build);
    h.update(model_dir);
    var b: [1024]u8 = undefined;
    const cfg_path = try std.fmt.bufPrint(&b, "{s}/config.json", .{model_dir});
    const cfg = Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(16 << 20)) catch "";
    defer if (cfg.len > 0) gpa.free(cfg);
    h.update(cfg);
    return h.final();
}

/// Opens `<base>/<fingerprint>` (made if missing), indexes the conversations in it and removes the other builds'
/// directories under `base`. Null when `cap` is 0 (off) or the directory cannot be made.
pub fn open(io: Io, base: []const u8, fp: u64, cap: u64) ?Disk {
    if (cap == 0) return null;
    return openOrErr(io, base, fp, cap) catch |e| {
        std.log.warn("[dsv41-disk] off: {s} ({s})", .{ base, @errorName(e) });
        return null;
    };
}

fn openOrErr(io: Io, base: []const u8, fp: u64, cap: u64) !Disk {
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, base);
    var name: [32]u8 = undefined;
    const mine = try std.fmt.bufPrint(&name, "{x:0>16}", .{fp});
    {
        var dir = try cwd.openDir(io, base, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |ent| {
            if (ent.kind == .directory and !std.mem.eql(u8, ent.name, mine)) dir.deleteTree(io, ent.name) catch {};
        }
    }
    var d: Disk = .{ .root = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ base, mine }), .cap = cap };
    errdefer d.deinit();
    try cwd.createDirPath(io, d.root);
    var dir = try cwd.openDir(io, d.root, .{ .iterate = true });
    defer dir.close(io);
    var it = dir.iterate();
    while (try it.next(io)) |ent| {
        if (ent.kind != .file or !std.mem.endsWith(u8, ent.name, ".ids")) continue;
        const id = std.fmt.parseInt(u64, ent.name[0 .. ent.name.len - 4], 10) catch continue;
        var b: [512]u8 = undefined;
        const side = d.path(&b, id, "ids") catch continue;
        const sc = readSidecar(io, side) catch continue;
        var b2: [512]u8 = undefined;
        const st_path = d.path(&b2, id, "safetensors") catch {
            gpa.free(sc.ids);
            continue;
        };
        const bytes = (cwd.statFile(io, st_path, .{}) catch {
            gpa.free(sc.ids);
            continue;
        }).size;
        d.tick += 1;
        try d.entries.append(gpa, .{ .id = id, .ids = sc.ids, .vision_key = sc.vision_key, .bytes = bytes, .tick = d.tick });
        d.next_id = @max(d.next_id, id + 1);
    }
    std.log.info("[dsv41-disk] {s}: {d} conversation(s), cap {d} MB", .{ d.root, d.entries.items.len, cap >> 20 });
    return d;
}
