//! The pack identity a peer prefix-cache transfer stamps and checks, hashed on its own thread.

const std = @import("std");
const builtin = @import("builtin");
const io_util = @import("../../../sdk/util/io_util.zig");
const log = @import("../../../sdk/util/log.zig");
const gguf_files = @import("../../../sdk/quant/gguf/files.zig");
const env = @import("../../../env.zig");

const chunk_bytes = 8 * 1024 * 1024;
const f_nocache: c_int = 48;
const qos_class_utility: c_uint = 0x11;

extern "c" fn pthread_set_qos_class_self_np(qos: c_uint, relative_priority: c_int) c_int;

pub var hold_for_test: std.atomic.Value(bool) = .init(false);
pub var parked_for_test: std.atomic.Value(bool) = .init(false);

/// Holds every hash at its next block; `waitParkedForTest` returns once one is parked there.
pub fn holdForTest() void {
    parked_for_test.store(false, .release);
    hold_for_test.store(true, .release);
}

pub fn waitParkedForTest(io: std.Io) void {
    while (!parked_for_test.load(.acquire)) io.sleep(.fromMilliseconds(1), .awake) catch {};
}

pub const Status = union(enum) { pending, unsupported, ready: u64 };

const State = enum(u8) { pending, ready, failed, cancelled };

/// The identity hashes all weight content in the background so peers can verify pack equality before importing.
pub const Job = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []u8,
    thread: std.Thread = undefined,
    cancel: std.atomic.Value(bool) = .init(false),
    state: std.atomic.Value(State) = .init(.pending),
    id: u64 = 0,

    fn run(job: *Job) void {
        if (comptime builtin.os.tag == .macos) _ = pthread_set_qos_class_self_np(qos_class_utility, 0);
        const t0 = io_util.nowUsMonotonic(job.io);
        var bytes_read: u64 = 0;
        const id = hashPack(job, &bytes_read) catch |err| {
            if (err == error.Canceled) {
                job.state.store(.cancelled, .release);
                return;
            }
            log.warn("[hot-cache] pack identity unavailable for {s}: {s}\n", .{ job.path, @errorName(err) });
            job.state.store(.failed, .release);
            return;
        };
        job.id = id;
        job.state.store(.ready, .release);
        const ms = @max(@divTrunc(io_util.nowUsMonotonic(job.io) - t0, 1000), 0);
        log.info("[hot-cache] pack identity: {d} bytes read ({d} ms)\n", .{ bytes_read, ms });
    }
};

/// Starts hashing `path` for a freshly loaded model. The inference thread owns `slot`.
pub fn start(io: std.Io, allocator: std.mem.Allocator, slot: *?*Job, path: []const u8) !void {
    drop(slot);
    const job = try allocator.create(Job);
    errdefer allocator.destroy(job);
    job.* = .{ .allocator = allocator, .io = io, .path = try allocator.dupe(u8, path) };
    errdefer allocator.free(job.path);
    job.thread = try std.Thread.spawn(.{}, Job.run, .{job});
    slot.* = job;
}

/// Never blocks. A load that started no hash, whose hash failed, or that runs replaced kernel sources
/// (MLX_SERVE_KERNEL_DIR) cannot transfer its cache.
pub fn status(job: ?*const Job) Status {
    if (env.present(.MLX_SERVE_KERNEL_DIR)) return .unsupported;
    const j = job orelse return .unsupported;
    return switch (j.state.load(.acquire)) {
        .pending => .pending,
        .ready => .{ .ready = j.id },
        .failed, .cancelled => .unsupported,
    };
}

/// Cancels the hash, which checks between blocks, and joins its thread before freeing what it reads.
pub fn drop(slot: *?*Job) void {
    const job = slot.* orelse return;
    slot.* = null;
    job.cancel.store(true, .release);
    job.thread.join();
    job.allocator.free(job.path);
    job.allocator.destroy(job);
}

pub fn waitReady(io: std.Io, job: ?*const Job) !u64 {
    while (true) {
        switch (status(job)) {
            .ready => |id| return id,
            .unsupported => return error.PackIdentityUnsupported,
            .pending => io.sleep(.fromMilliseconds(1), .awake) catch {},
        }
    }
}

const Hasher = struct {
    job: *Job,
    h: std.hash.Wyhash = .init(0x7061_636b_6964_656e),
    buf: []u8,
    bytes_read: u64 = 0,

    fn file(self: *Hasher, dir: std.Io.Dir, sub_path: []const u8, name: []const u8) !void {
        const io = self.job.io;
        const f = try dir.openFile(io, sub_path, .{});
        defer f.close(io);
        if (comptime builtin.os.tag == .macos) _ = std.c.fcntl(f.handle, f_nocache, @as(c_int, 1));
        const st = try f.stat(io);
        if (st.kind != .file) return error.NotARegularFile;
        const size = st.size;
        self.h.update(name);
        std.hash.autoHash(&self.h, size);
        var off: u64 = 0;
        while (off < size) {
            while (hold_for_test.load(.acquire) and !self.job.cancel.load(.acquire)) {
                parked_for_test.store(true, .release);
                io.sleep(.fromMilliseconds(1), .awake) catch {};
            }
            if (self.job.cancel.load(.acquire)) return error.Canceled;
            const n = try f.readPositionalAll(io, self.buf, off);
            if (n == 0) return error.UnexpectedEndOfFile;
            self.h.update(self.buf[0..n]);
            off += n;
        }
        self.bytes_read += off;
    }
};

/// The identity covers every byte the loader reads, for each way a model can be registered: a folder (recursively),
/// a file (its folder's direct children), and the DSpark/MTP stage GGUF the loader finds beside or above a GGUF.
/// The GGUF the loader selects (`resolveGgufFile`) and the stage it picks are named in it, so a sibling choice differs.
/// A folder holding several quantizations of one model hashes all of them, so two boxes must hold the same set to
/// exchange entries: a refusal, not an error.
fn hashPack(job: *Job, bytes_read: *u64) !u64 {
    const io = job.io;
    const buf = try job.allocator.alloc(u8, chunk_bytes);
    defer job.allocator.free(buf);
    var hasher: Hasher = .{ .job = job, .buf = buf };
    const cwd = std.Io.Dir.cwd();
    const st = try cwd.statFile(io, job.path, .{});
    const recursive = switch (st.kind) {
        .file => false,
        .directory => true,
        else => return error.UnsupportedPackPath,
    };
    const dir_path = if (recursive) job.path else std.fs.path.dirname(job.path) orelse return error.UnsupportedPackPath;
    var dir = try cwd.openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    const model = gguf_files.resolveGgufFile(io, job.allocator, job.path) catch null;
    defer if (model) |m| job.allocator.free(m);
    if (model) |m| {
        hasher.h.update("selected-gguf\x00");
        hasher.h.update(relativeTo(dir_path, m));
        hasher.h.update("\x00");
    }
    hasher.file(dir, "config.json", "config.json") catch |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    };
    const paths = try packPaths(io, job.allocator, dir, recursive);
    defer {
        for (paths) |p| job.allocator.free(p);
        job.allocator.free(paths);
    }
    for (paths) |p| try hasher.file(dir, p, p);
    if (model) |m| if (gguf_files.findMtpSidecar(io, job.allocator, m)) |stage| {
        defer job.allocator.free(stage);
        const above = std.fs.path.dirname(dir_path) orelse dir_path;
        try hasher.file(cwd, stage, under(dir_path, stage) orelse relativeTo(above, stage));
    };
    bytes_read.* = hasher.bytes_read;
    return hasher.h.final();
}

fn under(base: []const u8, path: []const u8) ?[]const u8 {
    if (path.len > base.len and std.mem.startsWith(u8, path, base) and std.fs.path.isSep(path[base.len])) return path[base.len + 1 ..];
    return null;
}

fn relativeTo(base: []const u8, path: []const u8) []const u8 {
    return under(base, path) orelse std.fs.path.basename(path);
}

fn packPaths(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, recursive: bool) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |p| allocator.free(p);
        out.deinit(allocator);
    }
    if (recursive) {
        var walker = try dir.walk(allocator);
        defer walker.deinit();
        while (try walker.next(io)) |e| try appendPackFile(allocator, &out, e.kind, e.basename, e.path);
    } else {
        var it = dir.iterate();
        while (try it.next(io)) |e| try appendPackFile(allocator, &out, e.kind, e.name, e.name);
    }
    std.mem.sort([]u8, out.items, {}, pathLess);
    return out.toOwnedSlice(allocator);
}

fn appendPackFile(allocator: std.mem.Allocator, out: *std.ArrayList([]u8), kind: std.Io.File.Kind, basename: []const u8, path: []const u8) !void {
    if (kind != .file and kind != .sym_link) return;
    if (!isPackFile(basename)) return;
    const p = try allocator.dupe(u8, path);
    errdefer allocator.free(p);
    try out.append(allocator, p);
}

fn isPackFile(name: []const u8) bool {
    inline for (.{ ".safetensors", ".bin", ".gguf" }) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return true;
    }
    inline for (.{ "model.safetensors.index.json", "engram-manifest.json", "engram-token-map.u32", "engram_token_map.json" }) |exact| {
        if (std.mem.eql(u8, name, exact)) return true;
    }
    return false;
}

fn pathLess(_: void, a: []u8, b: []u8) bool {
    return std.mem.lessThan(u8, a, b);
}

// ── tests ──

const testing = std.testing;

const TestDir = struct {
    tmp: testing.TmpDir,
    buf: [std.fs.max_path_bytes]u8 = undefined,
    len: usize = 0,

    fn init() !TestDir {
        var d: TestDir = .{ .tmp = testing.tmpDir(.{ .iterate = true }) };
        d.len = try d.tmp.dir.realPath(testing.io, &d.buf);
        return d;
    }

    fn path(d: *const TestDir) []const u8 {
        return d.buf[0..d.len];
    }

    fn write(d: *TestDir, name: []const u8, size: usize, flip: ?usize) !void {
        const w = try testing.allocator.alloc(u8, size);
        defer testing.allocator.free(w);
        for (w, 0..) |*b, i| b.* = @truncate(i *% 31);
        if (flip) |at| w[at] ^= 0x5a;
        try d.tmp.dir.writeFile(testing.io, .{ .sub_path = name, .data = w });
    }

    fn sub(d: *const TestDir, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ d.path(), name });
    }
};

fn identityOf(path: []const u8) !u64 {
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, path);
    return waitReady(testing.io, slot);
}

fn safetensorsPack(flip: ?usize) !TestDir {
    var d = try TestDir.init();
    errdefer d.tmp.cleanup();
    try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"x\"}" });
    try d.write("model.safetensors", 5 * 1024 * 1024, flip);
    return d;
}

test "pack identity: folder of safetensors: two 5 MiB packs that differ only at byte 65536 differ" {
    var a = try safetensorsPack(null);
    defer a.tmp.cleanup();
    var b = try safetensorsPack(65536);
    defer b.tmp.cleanup();
    try testing.expect(try identityOf(a.path()) != try identityOf(b.path()));
}

test "pack identity: folder of GGUF: config.json and different .gguf files differ" {
    var dirs: [2]TestDir = .{ try TestDir.init(), try TestDir.init() };
    defer for (&dirs) |*d| d.tmp.cleanup();
    var ids: [2]u64 = undefined;
    for (&dirs, &ids, 0..) |*d, *id, i| {
        try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"x\"}" });
        try d.write("a.gguf", 4096, null);
        try d.write("b.gguf", 4096, if (i == 0) null else 100);
        id.* = try identityOf(d.path());
    }
    try testing.expect(ids[0] != ids[1]);
}

fn splitPack(second_flip: ?usize, companion: []const u8) !TestDir {
    var d = try TestDir.init();
    errdefer d.tmp.cleanup();
    try d.write("model-00001-of-00002.gguf", 4096, null);
    try d.write(companion, 4096, second_flip);
    return d;
}

test "pack identity: split GGUF by first shard: a different second shard differs" {
    var a = try splitPack(null, "model-00002-of-00002.gguf");
    defer a.tmp.cleanup();
    var b = try splitPack(9, "model-00002-of-00002.gguf");
    defer b.tmp.cleanup();
    const first_a = try a.sub("model-00001-of-00002.gguf");
    defer testing.allocator.free(first_a);
    const first_b = try b.sub("model-00001-of-00002.gguf");
    defer testing.allocator.free(first_b);
    try testing.expect(try identityOf(first_a) != try identityOf(first_b));
}

test "pack identity: GGUF with an engram file beside it: a different engram file differs" {
    var a = try splitPack(null, "engram.gguf");
    defer a.tmp.cleanup();
    var b = try splitPack(9, "engram.gguf");
    defer b.tmp.cleanup();
    const first_a = try a.sub("model-00001-of-00002.gguf");
    defer testing.allocator.free(first_a);
    const first_b = try b.sub("model-00001-of-00002.gguf");
    defer testing.allocator.free(first_b);
    try testing.expect(try identityOf(first_a) != try identityOf(first_b));
}

test "pack identity: single GGUF file: its bytes decide it, and folders below it are not walked" {
    var a = try TestDir.init();
    defer a.tmp.cleanup();
    var b = try TestDir.init();
    defer b.tmp.cleanup();
    try a.write("model.gguf", 4096, null);
    try b.write("model.gguf", 4096, 3);
    const file_a = try a.sub("model.gguf");
    defer testing.allocator.free(file_a);
    const file_b = try b.sub("model.gguf");
    defer testing.allocator.free(file_b);
    const before = try identityOf(file_a);
    try testing.expect(before != try identityOf(file_b));
    try a.tmp.dir.createDirPath(testing.io, "nested");
    try a.write("nested/other.gguf", 4096, null);
    try testing.expectEqual(before, try identityOf(file_a));
}

test "pack identity: single GGUF file: two GGUFs in one folder registered by file differ" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    try d.write("a.gguf", 4096, null);
    try d.write("b.gguf", 4096, null);
    const a = try d.sub("a.gguf");
    defer testing.allocator.free(a);
    const b = try d.sub("b.gguf");
    defer testing.allocator.free(b);
    try testing.expect(try identityOf(a) != try identityOf(b));
}

test "pack identity: single GGUF file: a split GGUF by its first shard and a single GGUF in one folder differ" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    try d.write("m-00001-of-00002.gguf", 4096, null);
    try d.write("m-00002-of-00002.gguf", 4096, 5);
    try d.write("single.gguf", 4096, null);
    const shard = try d.sub("m-00001-of-00002.gguf");
    defer testing.allocator.free(shard);
    const single = try d.sub("single.gguf");
    defer testing.allocator.free(single);
    try testing.expect(try identityOf(shard) != try identityOf(single));
}

test "pack identity: folder of GGUF: it equals the file it resolves to, not that file's sibling" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    try d.write("a.gguf", 4096, null);
    try d.write("b.gguf", 4096, null);
    const a = try d.sub("a.gguf");
    defer testing.allocator.free(a);
    const b = try d.sub("b.gguf");
    defer testing.allocator.free(b);
    const folder = try identityOf(d.path());
    try testing.expectEqual(folder, try identityOf(a));
    try testing.expect(folder != try identityOf(b));
}

test "pack identity: GGUF with a stage file beside it: the MTP folder above it counts" {
    var dirs: [2]TestDir = .{ try TestDir.init(), try TestDir.init() };
    defer for (&dirs) |*d| d.tmp.cleanup();
    var ids: [2]u64 = undefined;
    for (&dirs, &ids, 0..) |*d, *id, i| {
        try d.tmp.dir.createDirPath(testing.io, "quant");
        try d.tmp.dir.createDirPath(testing.io, "MTP");
        try d.write("quant/model.gguf", 4096, null);
        try d.write("MTP/DSpark-support.gguf", 4096, if (i == 0) null else 11);
        const model = try d.sub("quant/model.gguf");
        defer testing.allocator.free(model);
        id.* = try identityOf(model);
    }
    try testing.expect(ids[0] != ids[1]);
}

test "pack identity: identical GGUF packs with a stage file in differently named folders are one pack" {
    var dirs: [2]TestDir = .{ try TestDir.init(), try TestDir.init() };
    defer for (&dirs) |*d| d.tmp.cleanup();
    const names = [_][]const u8{ "quant-a", "quant-b" };
    var ids: [2]u64 = undefined;
    for (&dirs, &ids, names) |*d, *id, folder| {
        try d.tmp.dir.createDirPath(testing.io, folder);
        const model_rel = try std.fmt.allocPrint(testing.allocator, "{s}/model.gguf", .{folder});
        defer testing.allocator.free(model_rel);
        const stage_rel = try std.fmt.allocPrint(testing.allocator, "{s}/DSpark-support.gguf", .{folder});
        defer testing.allocator.free(stage_rel);
        try d.write(model_rel, 4096, null);
        try d.write(stage_rel, 4096, null);
        const model = try d.sub(model_rel);
        defer testing.allocator.free(model);
        id.* = try identityOf(model);
    }
    try testing.expectEqual(ids[0], ids[1]);
}

test "pack identity: folder of safetensors: the engram token map counts" {
    var a = try safetensorsPack(null);
    defer a.tmp.cleanup();
    var b = try safetensorsPack(null);
    defer b.tmp.cleanup();
    try a.tmp.dir.writeFile(testing.io, .{ .sub_path = "engram-token-map.u32", .data = "\x01\x00\x00\x00" });
    try b.tmp.dir.writeFile(testing.io, .{ .sub_path = "engram-token-map.u32", .data = "\x02\x00\x00\x00" });
    try testing.expect(try identityOf(a.path()) != try identityOf(b.path()));
}

test "pack identity: a path that does not open is unsupported" {
    var d = try TestDir.init();
    defer d.tmp.cleanup();
    const missing = try d.sub("missing");
    defer testing.allocator.free(missing);
    try testing.expectError(error.PackIdentityUnsupported, identityOf(missing));
}

test "pack identity: a second read of the status touches no file" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    const first = try waitReady(testing.io, slot);
    try d.tmp.dir.deleteFile(testing.io, "config.json");
    try d.tmp.dir.deleteFile(testing.io, "model.safetensors");
    try testing.expectEqual(Status{ .ready = first }, status(slot));
}

test "pack identity: it stays pending while the test seam holds it" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    holdForTest();
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    waitParkedForTest(testing.io);
    try testing.expectEqual(Status.pending, status(slot));
    hold_for_test.store(false, .release);
    _ = try waitReady(testing.io, slot);
}

test "pack identity: dropping a job mid-hash joins its thread and frees it" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    holdForTest();
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    try start(testing.io, testing.allocator, &slot, d.path());
    waitParkedForTest(testing.io);
    drop(&slot);
    try testing.expectEqual(@as(?*Job, null), slot);
    try testing.expectEqual(Status.unsupported, status(slot));
}

test "pack identity: a file that shrinks while it is read fails the hash" {
    var d = try safetensorsPack(null);
    defer d.tmp.cleanup();
    holdForTest();
    defer hold_for_test.store(false, .release);
    var slot: ?*Job = null;
    defer drop(&slot);
    try start(testing.io, testing.allocator, &slot, d.path());
    waitParkedForTest(testing.io);
    try d.tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = "" });
    hold_for_test.store(false, .release);
    try testing.expectError(error.PackIdentityUnsupported, waitReady(testing.io, slot));
}
