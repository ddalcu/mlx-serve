//! Zero-copy GGUF reader: the file is mmapped, metadata strings, arrays and
//! tensor data are all slices into the map.
const std = @import("std");
const builtin = @import("builtin");
const env = @import("../../../env.zig");
const quants = @import("quants.zig");
const GgmlType = quants.GgmlType;

pub const Error = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    BadValueType,
    BadTensor,
    DuplicateKey,
} || std.mem.Allocator.Error;

const ValueType = enum(u32) { u8, i8, u16, i16, u32, i32, f32, bool, string, array, u64, i64, f64, _ };

fn scalarSize(t: ValueType) ?usize {
    return switch (t) {
        .u8, .i8, .bool => 1,
        .u16, .i16 => 2,
        .u32, .i32, .f32 => 4,
        .u64, .i64, .f64 => 8,
        else => null,
    };
}

pub const Array = struct {
    elem_type: ValueType,
    len: usize,
    /// Raw payload. Strings are (u64 len, bytes) back to back, walk them with `strings()`.
    bytes: []const u8,

    pub fn strings(self: Array) StringIterator {
        std.debug.assert(self.elem_type == .string);
        return .{ .r = .{ .buf = self.bytes }, .left = self.len };
    }

    pub fn boolean(self: Array, i: usize) bool {
        std.debug.assert(self.elem_type == .bool);
        return self.bytes[i] != 0;
    }

    pub fn float(self: Array, i: usize) f64 {
        const sz = scalarSize(self.elem_type).?;
        var r = Reader{ .buf = self.bytes[i * sz ..] };
        return switch (r.scalar(self.elem_type) catch unreachable) {
            .float => |v| v,
            else => unreachable,
        };
    }

    /// Scalar element as i64 (token types, eos id lists, ...).
    pub fn int(self: Array, i: usize) i64 {
        const sz = scalarSize(self.elem_type).?;
        var r = Reader{ .buf = self.bytes[i * sz ..] };
        return (r.scalar(self.elem_type) catch unreachable).asInt().?;
    }
};

pub const StringIterator = struct {
    r: Reader,
    left: usize,

    pub fn next(self: *StringIterator) ?[]const u8 {
        if (self.left == 0) return null;
        self.left -= 1;
        return self.r.string() catch unreachable; // validated at parse
    }
};

pub const Value = union(enum) {
    uint: u64,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,
    array: Array,

    pub fn asInt(self: Value) ?i64 {
        return switch (self) {
            .uint => |v| std.math.cast(i64, v),
            .int => |v| v,
            else => null,
        };
    }
};

pub const Tensor = struct {
    /// ggml order: dims[0] is the row length (in_features for a linear).
    dims: [4]u64,
    n_dims: u32,
    ty: GgmlType,
    data: []const u8,

    pub fn elems(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.dims[0..self.n_dims]) |d| n *= d;
        return n;
    }
};

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: u64) Error![]const u8 {
        if (n > self.buf.len - self.pos) return error.Truncated;
        const s = self.buf[self.pos..][0..@intCast(n)];
        self.pos += s.len;
        return s;
    }

    fn num(self: *Reader, comptime T: type) Error!T {
        const bytes = try self.take(@sizeOf(T));
        return @bitCast(std.mem.readInt(@Int(.unsigned, @bitSizeOf(T)), bytes[0..@sizeOf(T)], .little));
    }

    fn string(self: *Reader) Error![]const u8 {
        return self.take(try self.num(u64));
    }

    fn scalar(self: *Reader, t: ValueType) Error!Value {
        return switch (t) {
            .u8 => .{ .uint = try self.num(u8) },
            .u16 => .{ .uint = try self.num(u16) },
            .u32 => .{ .uint = try self.num(u32) },
            .u64 => .{ .uint = try self.num(u64) },
            .i8 => .{ .int = try self.num(i8) },
            .i16 => .{ .int = try self.num(i16) },
            .i32 => .{ .int = try self.num(i32) },
            .i64 => .{ .int = try self.num(i64) },
            .f32 => .{ .float = try self.num(f32) },
            .f64 => .{ .float = try self.num(f64) },
            .bool => .{ .bool = (try self.num(u8)) != 0 },
            else => error.BadValueType,
        };
    }

    fn value(self: *Reader) Error!Value {
        const t: ValueType = @fromBackingInt(@intCast(try self.num(u32)));
        switch (t) {
            .string => return .{ .string = try self.string() },
            .array => {
                const et: ValueType = @fromBackingInt(@intCast(try self.num(u32)));
                const len = std.math.cast(usize, try self.num(u64)) orelse return error.Truncated;
                const start = self.pos;
                if (et == .string) {
                    for (0..len) |_| _ = try self.string();
                } else {
                    const sz = scalarSize(et) orelse return error.BadValueType;
                    _ = try self.take(std.math.mul(u64, sz, len) catch return error.Truncated);
                }
                return .{ .array = .{ .elem_type = et, .len = len, .bytes = self.buf[start..self.pos] } };
            },
            else => return self.scalar(t),
        }
    }
};

pub const File = struct {
    arena: std.heap.ArenaAllocator,
    /// Non-null when we own an mmap (open), null for parse() over caller bytes.
    map: ?[]align(std.heap.page_size_min) const u8 = null,
    version: u32,
    kv: std.StringHashMapUnmanaged(Value) = .empty,
    tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty,
    /// The other shards of a split set (`split.count` > 1), whose tensors join `tensors`.
    shards: []const []align(std.heap.page_size_min) const u8 = &.{},
    /// Every mapped file's path, `map`'s first (empty for `parse`).
    paths: []const []const u8 = &.{},

    /// A split set's first shard (`<stem>-00001-of-0000N.gguf`) opens the whole set.
    pub fn open(allocator: std.mem.Allocator, path: []const u8) !File {
        if (comptime builtin.os.tag == .windows) return error.Unsupported; // mmap-based; the GGUF engine is Metal-only
        const map = try mapFile(path);
        var f = parse(allocator, map) catch |e| {
            std.posix.munmap(map);
            return e;
        };
        f.map = map;
        errdefer f.deinit();
        try f.openShards(allocator, path);
        return f;
    }

    pub const Location = struct { path: []const u8, offset: usize, tensor: Tensor };

    /// The file holding tensor `name` and its data's byte offset there (an mmapped file only).
    pub fn locate(self: *const File, name: []const u8) ?Location {
        const t = self.tensors.get(name) orelse return null;
        const at = @intFromPtr(t.data.ptr);
        for (self.paths, 0..) |path, i| {
            const m = if (i == 0) self.map orelse return null else self.shards[i - 1];
            const base = @intFromPtr(m.ptr);
            if (at >= base and at + t.data.len <= base + m.len) return .{ .path = path, .offset = at - base, .tensor = t };
        }
        return null;
    }

    /// Tensor `name`'s bytes read past the page cache (`F_NOCACHE`); null off macOS or for an unmapped file. macOS
    /// keeps a read file's pages (no madvise or msync drops them) and compresses the loader's copies first: a set
    /// that fills most of RAM swapped at load when the loader read the mapping. The read is page-aligned (an
    /// unaligned one went through the cache); `data` lies inside `buf`, which the caller frees.
    pub const Uncached = struct { buf: []align(std.heap.page_size_max) u8, data: []const u8 };

    pub fn readUncached(self: *const File, allocator: std.mem.Allocator, name: []const u8) !?Uncached {
        if (comptime !builtin.os.tag.isDarwin()) return null;
        const at = self.locate(name) orelse return null;
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (at.path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..at.path.len], at.path);
        pbuf[at.path.len] = 0;
        const fd = std.c.open(pbuf[0..at.path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd);
        _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
        const page = std.heap.page_size_max;
        const lo = std.mem.alignBackward(usize, at.offset, page);
        const hi = std.mem.alignForward(usize, at.offset + at.tensor.data.len, page);
        const buf = try allocator.alignedAlloc(u8, .fromByteUnits(page), hi - lo);
        errdefer allocator.free(buf);
        var done: usize = 0;
        while (done < buf.len) {
            const n = std.c.pread(fd, buf.ptr + done, @min(buf.len - done, 64 << 20), @intCast(lo + done));
            if (n < 0) return error.ReadFailed;
            if (n == 0) break; // the file's last page
            done += @intCast(n);
        }
        if (done < at.offset - lo + at.tensor.data.len) return error.ReadFailed;
        return .{ .buf = buf, .data = buf[at.offset - lo ..][0..at.tensor.data.len] };
    }

    fn mapFile(path: []const u8) ![]align(std.heap.page_size_min) const u8 {
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        if (path.len >= pbuf.len) return error.NameTooLong;
        @memcpy(pbuf[0..path.len], path);
        pbuf[path.len] = 0;
        const fd = std.c.open(pbuf[0..path.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.FileNotFound;
        defer _ = std.c.close(fd); // the mapping outlives the fd
        // lseek, not fstat: std.c.fstat is void on Linux in Zig 0.17.
        const size = std.c.lseek(fd, 0, std.c.SEEK.END);
        if (size < 0) return error.StatFailed;
        return std.posix.mmap(null, @intCast(size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    }

    fn openShards(self: *File, allocator: std.mem.Allocator, path: []const u8) !void {
        const a = self.arena.allocator();
        const count = std.math.cast(u32, self.getInt("split.count") orelse 1) orelse return error.BadSplit;
        const paths = try a.alloc([]const u8, @max(count, 1));
        paths[0] = try a.dupe(u8, path);
        self.paths = paths[0..1];
        if (count <= 1) return;
        if ((self.getInt("split.no") orelse 0) != 0) return error.BadSplit;
        var suffix_buf: [32]u8 = undefined;
        const suffix = std.fmt.bufPrint(&suffix_buf, "-00001-of-{d:0>5}.gguf", .{count}) catch return error.BadSplit;
        if (!std.mem.endsWith(u8, path, suffix)) return error.BadSplit;
        const stem = path[0 .. path.len - suffix.len];
        const maps = try a.alloc([]align(std.heap.page_size_min) const u8, count - 1);
        for (maps, 2..) |*m, no| {
            const shard_path = try std.fmt.allocPrint(a, "{s}-{d:0>5}-of-{d:0>5}.gguf", .{ stem, no, count });
            m.* = try mapFile(shard_path);
            self.shards = maps[0 .. no - 1];
            paths[no - 1] = shard_path;
            self.paths = paths[0..no];
            var part = try parse(allocator, m.*);
            defer part.deinit();
            for (part.tensors.keys(), part.tensors.values()) |name, t| {
                const gop = try self.tensors.getOrPut(a, name);
                if (gop.found_existing) return error.DuplicateKey;
                gop.value_ptr.* = t;
            }
        }
    }

    /// `bytes` must outlive the File.
    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) Error!File {
        var f = File{ .arena = std.heap.ArenaAllocator.init(allocator), .version = 0 };
        errdefer f.arena.deinit();
        const a = f.arena.allocator();

        var r = Reader{ .buf = bytes };
        if (!std.mem.eql(u8, try r.take(4), "GGUF")) return error.BadMagic;
        f.version = try r.num(u32);
        if (f.version < 2 or f.version > 3) return error.UnsupportedVersion;
        const n_tensors = try r.num(u64);
        const n_kv = try r.num(u64);

        for (0..n_kv) |_| {
            const key = try r.string();
            const gop = try f.kv.getOrPut(a, key);
            if (gop.found_existing) return error.DuplicateKey;
            gop.value_ptr.* = try r.value();
        }

        const Pending = struct { name: []const u8, t: Tensor, offset: u64 };
        const pending = try a.alloc(Pending, std.math.cast(usize, n_tensors) orelse return error.Truncated);
        for (pending) |*p| {
            p.name = try r.string();
            p.t = .{ .dims = @splat(1), .n_dims = try r.num(u32), .ty = undefined, .data = &.{} };
            if (p.t.n_dims > 4) return error.BadTensor;
            for (p.t.dims[0..p.t.n_dims]) |*d| d.* = try r.num(u64);
            p.t.ty = @fromBackingInt(@intCast(try r.num(u32)));
            p.offset = try r.num(u64);
        }

        const alignment: u64 = if (f.kv.get("general.alignment")) |v| @intCast(v.asInt() orelse 32) else 32;
        if (alignment == 0) return error.BadTensor;
        const data_start = (r.pos + alignment - 1) / alignment * alignment;

        for (pending) |p| {
            var t = p.t;
            const start = std.math.add(u64, data_start, p.offset) catch return error.BadTensor;
            // Unknown types stay in the directory with empty data at their offset, so callers can report exactly which
            // type blocked the load, and `locate` one an arch reads in place itself (DeepSeek-V4.1's I8 Engram rows).
            if (t.ty.supported()) {
                if (t.dims[0] % t.ty.blockElems() != 0) return error.BadTensor;
                const size = t.elems() / t.ty.blockElems() * t.ty.blockBytes();
                if (start > bytes.len or size > bytes.len - start) return error.Truncated;
                t.data = bytes[@intCast(start)..][0..@intCast(size)];
            } else if (start <= bytes.len) t.data = bytes[@intCast(start)..][0..0];
            try f.tensors.put(a, p.name, t);
        }
        return f;
    }

    pub fn deinit(self: *File) void {
        for (self.shards) |m| std.posix.munmap(m);
        self.arena.deinit();
        if (self.map) |m| std.posix.munmap(m);
    }

    pub fn getString(self: *const File, key: []const u8) ?[]const u8 {
        const v = self.kv.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    pub fn getInt(self: *const File, key: []const u8) ?i64 {
        return (self.kv.get(key) orelse return null).asInt();
    }

    pub fn getFloat(self: *const File, key: []const u8) ?f64 {
        const v = self.kv.get(key) orelse return null;
        return if (v == .float) v.float else null;
    }

    pub fn getBool(self: *const File, key: []const u8) ?bool {
        const v = self.kv.get(key) orelse return null;
        return if (v == .bool) v.bool else null;
    }

    pub fn getArray(self: *const File, key: []const u8) ?Array {
        const v = self.kv.get(key) orelse return null;
        return if (v == .array) v.array else null;
    }

    /// First tensor whose type we have no kernels for, null when all are servable.
    pub fn firstUnsupportedTensor(self: *const File) ?[]const u8 {
        var it = self.tensors.iterator();
        while (it.next()) |e| if (!e.value_ptr.ty.supported()) return e.key_ptr.*;
        return null;
    }
};

const TestWriter = struct {
    buf: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator = std.testing.allocator,

    fn num(self: *TestWriter, v: anytype) !void {
        try self.buf.appendSlice(self.a, std.mem.asBytes(&v));
    }
    fn str(self: *TestWriter, s: []const u8) !void {
        try self.num(@as(u64, s.len));
        try self.buf.appendSlice(self.a, s);
    }
};

/// Two KVs (string + string array) and one Q8_0 tensor [32 x 2].
fn buildTestFile(w: *TestWriter) !void {
    try w.buf.appendSlice(w.a, "GGUF");
    try w.num(@as(u32, 3));
    try w.num(@as(u64, 1));
    try w.num(@as(u64, 2));
    try w.str("general.architecture");
    try w.num(@as(u32, 8));
    try w.str("qwen35");
    try w.str("tokenizer.ggml.tokens");
    try w.num(@as(u32, 9));
    try w.num(@as(u32, 8));
    try w.num(@as(u64, 2));
    try w.str("hello");
    try w.str("world");
    try w.str("blk.0.ffn_up.weight");
    try w.num(@as(u32, 2));
    try w.num(@as(u64, 32));
    try w.num(@as(u64, 2));
    try w.num(@as(u32, 8));
    try w.num(@as(u64, 0));
    while (w.buf.items.len % 32 != 0) try w.buf.append(w.a, 0);
    try w.buf.appendNTimes(w.a, 7, 2 * 34);
}

test "parse reads metadata, string arrays and aligned tensor data" {
    var w = TestWriter{};
    defer w.buf.deinit(w.a);
    try buildTestFile(&w);

    var f = try File.parse(std.testing.allocator, w.buf.items);
    defer f.deinit();
    try std.testing.expectEqualStrings("qwen35", f.getString("general.architecture").?);
    var it = f.getArray("tokenizer.ggml.tokens").?.strings();
    try std.testing.expectEqualStrings("hello", it.next().?);
    try std.testing.expectEqualStrings("world", it.next().?);
    try std.testing.expect(it.next() == null);

    const t = f.tensors.get("blk.0.ffn_up.weight").?;
    try std.testing.expectEqual(GgmlType.q8_0, t.ty);
    try std.testing.expectEqual(@as(u64, 64), t.elems());
    try std.testing.expectEqual(@as(usize, 68), t.data.len);
    try std.testing.expectEqual(@as(u8, 7), t.data[0]);
    try std.testing.expect(f.firstUnsupportedTensor() == null);
}

test "parse keeps a tensor of a type we do not serve at its offset, with no data" {
    var w = TestWriter{};
    defer w.buf.deinit(w.a);
    try w.buf.appendSlice(w.a, "GGUF");
    try w.num(@as(u32, 3));
    try w.num(@as(u64, 2));
    try w.num(@as(u64, 0));
    for ([_][]const u8{ "q", "raw" }, [_]u32{ 8, 24 }, [_]u64{ 0, 64 }) |name, ty, off| {
        try w.str(name);
        try w.num(@as(u32, 1));
        try w.num(@as(u64, 32));
        try w.num(ty);
        try w.num(off);
    }
    while (w.buf.items.len % 32 != 0) try w.buf.append(w.a, 0);
    const data_start = w.buf.items.len;
    try w.buf.appendNTimes(w.a, 9, 64 + 32);

    var f = try File.parse(std.testing.allocator, w.buf.items);
    defer f.deinit();
    const raw = f.tensors.get("raw").?;
    try std.testing.expectEqual(@as(usize, 0), raw.data.len);
    try std.testing.expectEqual(@intFromPtr(w.buf.items.ptr) + data_start + 64, @intFromPtr(raw.data.ptr));
    try std.testing.expectEqualStrings("raw", f.firstUnsupportedTensor().?);
}

test "parse rejects a file cut short anywhere" {
    var w = TestWriter{};
    defer w.buf.deinit(w.a);
    try buildTestFile(&w);
    for (0..w.buf.items.len) |n| {
        if (File.parse(std.testing.allocator, w.buf.items[0..n])) |*f| {
            @constCast(f).deinit();
            return error.TestExpectedError;
        } else |_| {}
    }
}

/// One shard of a split set: `split.no`/`split.count` and, past shard 1, one Q8_0 tensor `t<no>` [32 x 1].
fn buildShard(w: *TestWriter, no: u16, count: u16) !void {
    try w.buf.appendSlice(w.a, "GGUF");
    try w.num(@as(u32, 3));
    try w.num(@as(u64, @intFromBool(no > 0)));
    try w.num(@as(u64, 2));
    try w.str("split.no");
    try w.num(@as(u32, 2));
    try w.num(no);
    try w.str("split.count");
    try w.num(@as(u32, 2));
    try w.num(count);
    if (no > 0) {
        var name: [8]u8 = undefined;
        try w.str(std.fmt.bufPrint(&name, "t{d}", .{no}) catch unreachable);
        try w.num(@as(u32, 1));
        try w.num(@as(u64, 32));
        try w.num(@as(u32, 8));
        try w.num(@as(u64, 0));
    }
    while (w.buf.items.len % 32 != 0) try w.buf.append(w.a, 0);
    if (no > 0) try w.buf.appendNTimes(w.a, @intCast(no), 34);
}

test "open follows a split set from its first shard, and refuses a missing sibling" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    for (0..3) |i| {
        var w = TestWriter{};
        defer w.buf.deinit(w.a);
        try buildShard(&w, @intCast(i), 3);
        var name: [32]u8 = undefined;
        try tmp.dir.writeFile(io, .{ .sub_path = std.fmt.bufPrint(&name, "m-{d:0>5}-of-00003.gguf", .{i + 1}) catch unreachable, .data = w.buf.items });
    }
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const first = try std.fmt.allocPrint(std.testing.allocator, "{s}/m-00001-of-00003.gguf", .{root});
    defer std.testing.allocator.free(first);

    var f = try File.open(std.testing.allocator, first);
    defer f.deinit();
    try std.testing.expectEqual(@as(usize, 2), f.tensors.count());
    try std.testing.expectEqual(@as(u8, 2), f.tensors.get("t2").?.data[0]);
    try std.testing.expectEqual(@as(u8, 1), f.tensors.get("t1").?.data[0]);
    const at = f.locate("t2").?;
    try std.testing.expect(std.mem.endsWith(u8, at.path, "m-00003-of-00003.gguf"));
    try std.testing.expectEqual(@as(usize, 2), at.tensor.data[0]);
    try std.testing.expectEqual(@as(usize, 0), at.offset % 32);
    if (try f.readUncached(std.testing.allocator, "t2")) |r| {
        defer std.testing.allocator.free(r.buf);
        try std.testing.expectEqualSlices(u8, f.tensors.get("t2").?.data, r.data);
    }

    try tmp.dir.deleteFile(io, "m-00003-of-00003.gguf");
    try std.testing.expectError(error.FileNotFound, File.open(std.testing.allocator, first));
}

test "real model file parses (a real GGUF, env-gated)" {
    const path = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    var f = try File.open(std.testing.allocator, path);
    defer f.deinit();
    try std.testing.expect(f.getString("general.architecture").?.len > 0);
    try std.testing.expect(f.firstUnsupportedTensor() == null);
    try std.testing.expect(f.getArray("tokenizer.ggml.tokens").?.len > 10_000);

    // Last block of the last tensor must be readable (catches offset/size math).
    const t = f.tensors.values()[f.tensors.count() - 1];
    var y: [256]f32 = undefined;
    if (t.ty.isQuantized()) {
        const bb = t.ty.blockBytes();
        try quants.dequantize(t.ty, t.data[t.data.len - bb ..], y[0..t.ty.blockElems()]);
        for (y[0..t.ty.blockElems()]) |v| try std.testing.expect(std.math.isFinite(v));
    }
}
