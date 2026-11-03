//! A safetensors checkpoint's headers (8-byte length + JSON, never tensor data) and one tensor's bytes, for the
//! streamed packs' residents: shards from `model.safetensors.index.json` or one file.

const std = @import("std");

pub const max_rank = 6;

pub const Refusal = error{ IndexMissing, IndexSyntax, IndexMismatch, ShardMissing, ShardHeader, TensorBytes, TensorMissing };
pub const Error = Refusal || std.mem.Allocator.Error;

/// Why a checkpoint was refused, for the one log line the caller writes.
pub const Diag = @import("sdk").Diag;

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// Safetensors element types (the header's `dtype` strings).
pub const StDtype = enum {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,

    pub fn size(d: StDtype) u64 {
        return switch (d) {
            .BOOL, .U8, .I8 => 1,
            .U16, .I16, .F16, .BF16 => 2,
            .U32, .I32, .F32 => 4,
            .U64, .I64, .F64 => 8,
        };
    }
};

pub const Checkpoint = struct {
    arena: std.heap.ArenaAllocator,
    dir: []const u8,
    shards: std.ArrayList(Shard) = .empty,
    tensors: std.array_hash_map.String(Tensor) = .empty,

    pub const Shard = struct { name: []const u8, file_size: u64, data_start: u64 };
    pub const Tensor = struct {
        shard: u16,
        dtype: StDtype,
        shape: [max_rank]u64,
        rank: u8,
        /// Absolute file offsets of the tensor bytes.
        begin: u64,
        end: u64,

        pub fn numel(t: Tensor) u64 {
            var n: u64 = 1;
            for (t.shape[0..t.rank]) |d| n *= d;
            return n;
        }
    };

    pub fn deinit(self: *Checkpoint) void {
        self.arena.deinit();
    }

    /// A sharded checkpoint: `model.safetensors.index.json` names the shard of
    /// every tensor; each shard's header must declare exactly the tensors the
    /// index assigns to it.
    pub fn openIndexed(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, diag: ?*Diag) Error!Checkpoint {
        var self: Checkpoint = .{ .arena = std.heap.ArenaAllocator.init(gpa), .dir = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        self.dir = try a.dupe(u8, dir);
        const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
        const text = std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(64 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.IndexMissing, "{s}: {s}", .{ index_path, @errorName(e) }),
        };
        const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
        const index = std.json.parseFromSliceLeaky(Index, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.IndexSyntax, "{s}: {s}", .{ index_path, @errorName(e) }),
        };
        const wm = index.weight_map.map;
        for (wm.values()) |shard_name| {
            for (self.shards.items) |s| {
                if (std.mem.eql(u8, s.name, shard_name)) break;
            } else try self.addShard(a, shard_name, diag);
        }
        for (wm.keys(), wm.values()) |name, shard_name| {
            const t = self.tensors.get(name) orelse return refuse(diag, error.IndexMismatch, "{s}: named by the index, absent from {s}", .{ name, shard_name });
            if (!std.mem.eql(u8, self.shards.items[t.shard].name, shard_name)) return refuse(diag, error.IndexMismatch, "{s}: index says {s}, found in {s}", .{ name, shard_name, self.shards.items[t.shard].name });
        }
        if (self.tensors.count() != wm.count()) {
            for (self.tensors.keys()) |name| {
                if (wm.get(name) == null) return refuse(diag, error.IndexMismatch, "{s}: in a shard, not in the index", .{name});
            }
        }
        return self;
    }

    /// One safetensors file (the Engram sidecar, a parity dump).
    pub fn openFile(gpa: std.mem.Allocator, path: []const u8, diag: ?*Diag) Error!Checkpoint {
        var self: Checkpoint = .{ .arena = std.heap.ArenaAllocator.init(gpa), .dir = undefined };
        errdefer self.deinit();
        const a = self.arena.allocator();
        self.dir = try a.dupe(u8, std.fs.path.dirname(path) orelse ".");
        try self.addShard(a, std.fs.path.basename(path), diag);
        return self;
    }

    pub fn shardPath(self: *const Checkpoint, a: std.mem.Allocator, shard: u16) ![:0]u8 {
        return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ self.dir, self.shards.items[shard].name }, 0);
    }

    fn addShard(self: *Checkpoint, a: std.mem.Allocator, name: []const u8, diag: ?*Diag) Error!void {
        if (self.shards.items.len >= std.math.maxInt(u16)) return refuse(diag, error.ShardHeader, "too many shards", .{});
        const shard: u16 = @intCast(self.shards.items.len);
        const owned = try a.dupe(u8, name);
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ self.dir, owned }, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
        if (fd < 0) return refuse(diag, error.ShardMissing, "{s}: cannot open", .{path});
        defer _ = std.c.close(fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(fd, &st) != 0) return refuse(diag, error.ShardMissing, "{s}: fstat failed", .{path});
        const file_size: u64 = @intCast(st.size);
        var lenb: [8]u8 = undefined;
        if (!preadAll(fd, &lenb, 0)) return refuse(diag, error.ShardHeader, "{s}: no header length", .{path});
        const n = std.mem.readInt(u64, &lenb, .little);
        if (n < 2 or n > max_header_bytes or 8 + n > file_size) return refuse(diag, error.ShardHeader, "{s}: header length {d} invalid for a {d}-byte file", .{ path, n, file_size });
        const json = try a.alloc(u8, @intCast(n));
        if (!preadAll(fd, json, 8)) return refuse(diag, error.ShardHeader, "{s}: short header read", .{path});
        const data_start = 8 + n;
        try self.shards.append(a, .{ .name = owned, .file_size = file_size, .data_start = data_start });

        const TensorJson = struct { dtype: []const u8 = "", shape: []const u64 = &.{}, data_offsets: []const u64 = &.{} };
        const hdr = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(TensorJson), a, json, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ShardHeader, "{s}: header JSON {s}", .{ path, @errorName(e) }),
        };
        for (hdr.map.keys(), hdr.map.values()) |tname, tj| {
            if (std.mem.eql(u8, tname, "__metadata__")) continue;
            const dtype = std.meta.stringToEnum(StDtype, tj.dtype) orelse return refuse(diag, error.ShardHeader, "{s}: {s} has dtype \"{s}\"", .{ path, tname, tj.dtype });
            if (tj.shape.len > max_rank or tj.data_offsets.len != 2) return refuse(diag, error.ShardHeader, "{s}: {s} has a bad shape or offsets", .{ path, tname });
            var t: Tensor = .{ .shard = shard, .dtype = dtype, .shape = @splat(0), .rank = @intCast(tj.shape.len), .begin = data_start + tj.data_offsets[0], .end = data_start + tj.data_offsets[1] };
            @memcpy(t.shape[0..t.rank], tj.shape);
            if (tj.data_offsets[0] > tj.data_offsets[1] or t.end > file_size) return refuse(diag, error.TensorBytes, "{s}: {s} data [{d}, {d}) outside the file", .{ path, tname, tj.data_offsets[0], tj.data_offsets[1] });
            if (t.end - t.begin != t.numel() * dtype.size()) return refuse(diag, error.TensorBytes, "{s}: {s} holds {d} bytes, its shape needs {d}", .{ path, tname, t.end - t.begin, t.numel() * dtype.size() });
            const gop = try self.tensors.getOrPut(a, tname);
            if (gop.found_existing) return refuse(diag, error.IndexMismatch, "{s}: {s} is declared twice", .{ path, tname });
            gop.value_ptr.* = t;
        }
    }
};

const max_header_bytes: u64 = 64 << 20;

fn preadAll(fd: std.c.fd_t, buf: []u8, offset: u64) bool {
    var done: usize = 0;
    while (done < buf.len) {
        const r = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(offset + done));
        if (r < 0) {
            if (std.c._errno().* == @backingInt(std.posix.E.INTR)) continue;
            return false;
        }
        if (r == 0) return false;
        done += @intCast(r);
    }
    return true;
}

/// Read one tensor's bytes (host-side; parity dumps and fixtures).
pub fn readTensor(a: std.mem.Allocator, ck: *const Checkpoint, name: []const u8) ![]u8 {
    const t = ck.tensors.get(name) orelse return error.TensorMissing;
    const path = try ck.shardPath(a, t.shard);
    defer a.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.ShardMissing;
    defer _ = std.c.close(fd);
    const buf = try a.alloc(u8, @intCast(t.end - t.begin));
    errdefer a.free(buf);
    if (!preadAll(fd, buf, t.begin)) return error.TensorBytes;
    return buf;
}
