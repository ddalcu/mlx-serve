//! Prism Bonsai packs ship no MTP head, but the trunk is Qwen3.8-27B's and the
//! base model's own head drafts for it (75% acceptance on code). At load, a
//! Hadamard pack of that geometry without a head gets `mtp.safetensors`
//! assembled from our Qwen3.8-27B pack: copied from a local install when one
//! sits beside it, else byte-range fetched from Hugging Face (~314 MB, only
//! the head's tensors). Best effort: any failure logs and the model serves
//! without MTP.
const std = @import("std");
const build_options = @import("build_options");
const model_mod = @import("model.zig");
const mtp_mod = @import("mtp.zig");
const log = @import("log.zig");

const DONOR_REPO = "ddalcu/Qwen3.8-27B-MLX-Serve-4bit";
const SIDECAR = "mtp.safetensors";
const INDEX = "model.safetensors.index.json";

/// The donor head fits this trunk: a Hadamard pack of Qwen3.8-27B's shape.
pub fn wanted(config: *const model_mod.ModelConfig) bool {
    return config.hadamard_block > 0 and config.hidden_size == 5120 and
        config.num_hidden_layers == 64 and config.vocab_size == 248320;
}

const Source = union(enum) {
    local: []const u8, // absolute donor dir
    remote,

    fn read(self: Source, allocator: std.mem.Allocator, io: std.Io, rel: []const u8, off: u64, len: u64) ![]u8 {
        switch (self) {
            .local => |dir| {
                const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, rel }, 0);
                defer allocator.free(path);
                const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
                if (fd < 0) return error.DonorRead;
                defer _ = std.c.close(fd);
                const buf = try allocator.alloc(u8, @intCast(len));
                errdefer allocator.free(buf);
                var done: usize = 0;
                while (done < buf.len) {
                    const got = std.c.pread(fd, buf[done..].ptr, buf.len - done, @intCast(off + done));
                    if (got <= 0) return error.DonorRead;
                    done += @intCast(got);
                }
                return buf;
            },
            .remote => {
                const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ DONOR_REPO, rel });
                defer allocator.free(url);
                const range = try std.fmt.allocPrint(allocator, "{d}-{d}", .{ off, off + len - 1 });
                defer allocator.free(range);
                const result = std.process.run(allocator, io, .{
                    .argv = &.{ "curl", "-fsSL", "--retry", "3", "--retry-delay", "2", "-r", range, url },
                    .stdout_limit = .limited(@intCast(len + 1)),
                }) catch return error.DonorFetch;
                defer allocator.free(result.stderr);
                errdefer allocator.free(result.stdout);
                switch (result.term) {
                    .exited => |code| if (code != 0) return error.DonorFetch,
                    else => return error.DonorFetch,
                }
                if (result.stdout.len != len) return error.DonorShortRead;
                return result.stdout;
            },
        }
    }

    fn readWhole(self: Source, allocator: std.mem.Allocator, io: std.Io, rel: []const u8) ![]u8 {
        switch (self) {
            .local => |dir| {
                const path = try std.fs.path.join(allocator, &.{ dir, rel });
                defer allocator.free(path);
                const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
                defer f.close(io);
                var rb: [8192]u8 = undefined;
                var rs = f.reader(io, &rb);
                return rs.interface.allocRemaining(allocator, .limited(16 * 1024 * 1024));
            },
            .remote => {
                const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ DONOR_REPO, rel });
                defer allocator.free(url);
                const result = std.process.run(allocator, io, .{
                    .argv = &.{ "curl", "-fsSL", "--retry", "3", "--retry-delay", "2", url },
                    .stdout_limit = .limited(16 * 1024 * 1024),
                }) catch return error.DonorFetch;
                defer allocator.free(result.stderr);
                errdefer allocator.free(result.stdout);
                switch (result.term) {
                    .exited => |code| if (code != 0) return error.DonorFetch,
                    else => return error.DonorFetch,
                }
                return result.stdout;
            },
        }
    }
};

const Tensor = struct {
    name: []const u8,
    shard: []const u8,
    dtype: []const u8,
    shape: []i64,
    off: u64, // absolute file offset in the shard
    len: u64,
};

fn fileExists(io: std.Io, path: []const u8) bool {
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

/// A donor install next to the model (same models root) or in the default root.
fn localDonor(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ?[]u8 {
    var candidates: [2]?[]u8 = .{ null, null };
    if (std.fs.path.dirname(model_dir)) |org| if (std.fs.path.dirname(org)) |root| {
        candidates[0] = std.fs.path.join(allocator, &.{ root, DONOR_REPO }) catch null;
    };
    if (std.c.getenv("HOME")) |home| {
        candidates[1] = std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ std.mem.span(home), DONOR_REPO }) catch null;
    }
    var found: ?[]u8 = null;
    for (candidates) |c| if (c) |dir| {
        const idx = std.fs.path.join(allocator, &.{ dir, INDEX }) catch {
            allocator.free(dir);
            continue;
        };
        defer allocator.free(idx);
        if (found == null and fileExists(io, idx)) found = dir else allocator.free(dir);
    };
    return found;
}

/// Writes `<model_dir>/mtp.safetensors` when `wanted` and the dir has no head.
pub fn ensure(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, config: *const model_mod.ModelConfig) void {
    if (!wanted(config) or mtp_mod.hasMtpHead(io, allocator, model_dir)) return;
    const donor = localDonor(allocator, io, model_dir);
    defer if (donor) |d| allocator.free(d);
    if (donor == null and build_options.mas) return;
    const src: Source = if (donor) |d| .{ .local = d } else .remote;
    log.info("[mtp] no head in this Hadamard pack; assembling one from {s} ({s}); the load waits on it\n", .{ DONOR_REPO, if (donor != null) "local install" else "Hugging Face, ~314 MB" });
    build(allocator, io, src, model_dir) catch |err| {
        log.warn("[mtp] head graft failed ({s}); serving without MTP\n", .{@errorName(err)});
        return;
    };
    log.info("[mtp] wrote {s}/{s}\n", .{ model_dir, SIDECAR });
}

fn build(allocator: std.mem.Allocator, io: std.Io, src: Source, model_dir: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const index_bytes = try src.readWhole(arena, io, INDEX);
    const index = try std.json.parseFromSliceLeaky(std.json.Value, arena, index_bytes, .{});
    const wmap = (index.object.get("weight_map") orelse return error.DonorIndex).object;

    var tensors = std.ArrayList(Tensor).empty;
    var shards = std.ArrayList([]const u8).empty;
    var it = wmap.iterator();
    while (it.next()) |kv| {
        if (std.mem.indexOf(u8, kv.key_ptr.*, ".mtp.") == null) continue;
        const shard = kv.value_ptr.string;
        for (shards.items) |have| {
            if (std.mem.eql(u8, have, shard)) break;
        } else try shards.append(arena, shard);
    }
    for (shards.items) |shard| {
        const len_bytes = try src.read(arena, io, shard, 0, 8);
        const hdr_len = std.mem.readInt(u64, len_bytes[0..8], .little);
        if (hdr_len == 0 or hdr_len > 64 * 1024 * 1024) return error.DonorHeader;
        const hdr_bytes = try src.read(arena, io, shard, 8, hdr_len);
        const hdr = try std.json.parseFromSliceLeaky(std.json.Value, arena, hdr_bytes, .{});
        var hit = hdr.object.iterator();
        while (hit.next()) |kv| {
            const name = kv.key_ptr.*;
            if (std.mem.indexOf(u8, name, ".mtp.") == null) continue;
            const e = kv.value_ptr.object;
            const offs = (e.get("data_offsets") orelse return error.DonorHeader).array.items;
            const shape_items = (e.get("shape") orelse return error.DonorHeader).array.items;
            const shape = try arena.alloc(i64, shape_items.len);
            for (shape_items, shape) |v, *d| d.* = v.integer;
            const a: u64 = @intCast(offs[0].integer);
            const b: u64 = @intCast(offs[1].integer);
            try tensors.append(arena, .{
                .name = name,
                .shard = shard,
                .dtype = (e.get("dtype") orelse return error.DonorHeader).string,
                .shape = shape,
                .off = 8 + hdr_len + a,
                .len = b - a,
            });
        }
    }
    if (tensors.items.len == 0) return error.DonorHasNoHead;

    // Header: offsets are the concatenation order, padded to 8 bytes.
    var header = std.ArrayList(u8).empty;
    try header.appendSlice(arena, "{\"__metadata__\":{\"format\":\"mlx\"}");
    var cursor: u64 = 0;
    for (tensors.items) |t| {
        try header.print(arena, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, i| try header.print(arena, "{s}{d}", .{ if (i == 0) "" else ",", d });
        try header.print(arena, "],\"data_offsets\":[{d},{d}]}}", .{ cursor, cursor + t.len });
        cursor += t.len;
    }
    try header.append(arena, '}');
    while (header.items.len % 8 != 0) try header.append(arena, ' ');

    // Per-process partial name: two loads (app + CLI) building at once must
    // not truncate each other's file; the rename is atomic, last one wins.
    const final_path = try std.fs.path.join(arena, &.{ model_dir, SIDECAR });
    const partial = try std.fmt.allocPrintSentinel(arena, "{s}.{d}.partial", .{ final_path, getpid() }, 0);
    const fd = std.c.open(partial, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.SidecarNotWritable;
    var closed = false;
    defer if (!closed) {
        _ = std.c.close(fd);
    };
    errdefer _ = std.c.unlink(partial);

    var len_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_le, header.items.len, .little);
    try writeAll(fd, &len_le);
    try writeAll(fd, header.items);
    for (tensors.items) |t| {
        const bytes = try src.read(allocator, io, t.shard, t.off, t.len);
        defer allocator.free(bytes);
        try writeAll(fd, bytes);
    }
    _ = std.c.close(fd);
    closed = true;
    try std.Io.Dir.renameAbsolute(partial, final_path, io);
}

extern "c" fn getpid() c_int;

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (n <= 0) return error.SidecarWrite;
        done += @intCast(n);
    }
}

test "mtp_graft: copies only the head's tensors into a valid safetensors file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);

    // donor: one shard holding a trunk tensor and two head tensors
    const hdr =
        \\{"model.layers.0.w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"language_model.mtp.fc.weight":{"dtype":"U8","shape":[2,3],"data_offsets":[4,10]},"language_model.mtp.norm.weight":{"dtype":"U8","shape":[2],"data_offsets":[10,12]}}
    ;
    const data = [_]u8{ 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 20, 21 };
    try tmp.dir.createDirPath(io, "donor");
    try tmp.dir.createDirPath(io, "model");
    {
        var f = try tmp.dir.createFile(io, "donor/s1.safetensors", .{});
        defer f.close(io);
        var wb: [256]u8 = undefined;
        var w = f.writer(io, &wb);
        var len_le: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_le, hdr.len, .little);
        try w.interface.writeAll(&len_le);
        try w.interface.writeAll(hdr);
        try w.interface.writeAll(&data);
        try w.interface.flush();
    }
    {
        var f = try tmp.dir.createFile(io, "donor/" ++ INDEX, .{});
        defer f.close(io);
        var wb: [256]u8 = undefined;
        var w = f.writer(io, &wb);
        try w.interface.writeAll(
            \\{"weight_map":{"model.layers.0.w":"s1.safetensors","language_model.mtp.fc.weight":"s1.safetensors","language_model.mtp.norm.weight":"s1.safetensors"}}
        );
        try w.interface.flush();
    }
    const donor = try std.fs.path.join(allocator, &.{ root, "donor" });
    defer allocator.free(donor);
    const model = try std.fs.path.join(allocator, &.{ root, "model" });
    defer allocator.free(model);
    try build(allocator, io, .{ .local = donor }, model);

    const out = try tmp.dir.readFileAlloc(io, "model/" ++ SIDECAR, allocator, .limited(4096));
    defer allocator.free(out);
    const hlen = std.mem.readInt(u64, out[0..8], .little);
    try std.testing.expect(hlen % 8 == 0);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, out[8 .. 8 + hlen], .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("model.layers.0.w") == null);
    const body = out[8 + hlen ..];
    for ([_]struct { []const u8, []const u8 }{ .{ "language_model.mtp.fc.weight", data[4..10] }, .{ "language_model.mtp.norm.weight", data[10..12] } }) |want| {
        const offs = parsed.value.object.get(want[0]).?.object.get("data_offsets").?.array.items;
        try std.testing.expectEqualSlices(u8, want[1], body[@intCast(offs[0].integer)..@intCast(offs[1].integer)]);
    }
}
