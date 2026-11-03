//! Tiny packs for tests, MLX-free: the files a plugin hands the conformance suite, a `PackDir` over them in memory,
//! and the safetensors bytes of a random-weight model.

const std = @import("std");
const common = @import("../../abi/common.zig");

const Str = common.Str;
const Verdict = common.Verdict;
const PackDir = common.PackDir;

pub const File = struct { name: []const u8, bytes: []const u8 };

/// A plugin's tiny pack: its files (a model's `config.json` among them), plus a media job's body or a quant block.
pub const Fixture = struct {
    files: []const File,
    /// A media plugin's request body.
    body: []const u8 = "{}",
    /// A media plugin's request inputs (`media.Request.inputs`), in order.
    inputs: []const []const u8 = &.{},
    /// A quant plugin's claimed block (JSON).
    block: []const u8 = "",

    pub fn file(self: Fixture, name: []const u8) ?[]const u8 {
        for (self.files) |f| if (std.mem.eql(u8, f.name, name)) return f.bytes;
        return null;
    }
};

/// A `PackDir` over files in memory; a folder exists when a file lies under it.
pub const MemDir = struct {
    files: []const File,

    pub fn packDir(self: *const MemDir) PackDir {
        return .{ .ctx = @constCast(self), .exists = exists, .read = read, .safetensors_header = header, .list = list, .file_bytes = fileBytes };
    }

    fn get(ctx: ?*anyopaque, rel: Str) ?[]const u8 {
        const self: *const MemDir = @ptrCast(@alignCast(ctx.?));
        for (self.files) |f| if (std.mem.eql(u8, f.name, rel.slice())) return f.bytes;
        return null;
    }
    fn exists(ctx: ?*anyopaque, rel: Str) callconv(.c) bool {
        if (get(ctx, rel) != null) return true;
        const self: *const MemDir = @ptrCast(@alignCast(ctx.?));
        const r = rel.slice();
        for (self.files) |f| if (f.name.len > r.len and std.mem.startsWith(u8, f.name, r) and f.name[r.len] == '/') return true;
        return false;
    }
    fn copy(bytes: []const u8, buf: [*]u8, cap: usize, len: *usize) Verdict {
        if (bytes.len > cap) return Verdict.refuse(.too_large, "file larger than the read bound");
        @memcpy(buf[0..bytes.len], bytes);
        len.* = bytes.len;
        return .ok;
    }
    fn read(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const bytes = get(ctx, rel) orelse return Verdict.refuse(.failed, "no such file in the pack");
        return copy(bytes, buf, cap, len);
    }
    fn header(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const bytes = get(ctx, rel) orelse return Verdict.refuse(.failed, "no such file in the pack");
        if (bytes.len < 8) return Verdict.refuse(.failed, "not a safetensors file");
        const n = std.mem.readInt(u64, bytes[0..8], .little);
        if (n > bytes.len - 8) return Verdict.refuse(.failed, "safetensors header past the end of the file");
        return copy(bytes[8..][0..@intCast(n)], buf, cap, len);
    }
    fn list(ctx: ?*anyopaque, rel: Str, buf: [*]u8, cap: usize, len: *usize) callconv(.c) Verdict {
        const self: *const MemDir = @ptrCast(@alignCast(ctx.?));
        const r = rel.slice();
        if (r.len > 0 and !exists(ctx, rel)) return Verdict.refuse(.failed, "no such folder in the pack");
        var n: usize = 0;
        for (self.files, 0..) |f, i| {
            if (r.len > 0 and !(f.name.len > r.len and std.mem.startsWith(u8, f.name, r) and f.name[r.len] == '/')) continue;
            const tail = f.name[if (r.len > 0) r.len + 1 else 0..];
            const slash = std.mem.indexOfScalar(u8, tail, '/');
            const entry = if (slash) |k| tail[0 .. k + 1] else tail;
            // A folder is listed once, at the first file under it.
            const seen = for (self.files[0..i]) |g| {
                if (r.len > 0 and !(g.name.len > r.len and std.mem.startsWith(u8, g.name, r) and g.name[r.len] == '/')) continue;
                if (std.mem.startsWith(u8, g.name[if (r.len > 0) r.len + 1 else 0..], entry) and slash != null) break true;
            } else false;
            if (seen) continue;
            if (n + entry.len + 1 > cap) return Verdict.refuse(.too_large, "listing larger than the read bound");
            @memcpy(buf[n..][0..entry.len], entry);
            buf[n + entry.len] = 0;
            n += entry.len + 1;
        }
        len.* = n;
        return .ok;
    }
    fn fileBytes(ctx: ?*anyopaque, rel: Str) callconv(.c) u64 {
        return if (get(ctx, rel)) |b| b.len else 0;
    }
};

/// One f32 tensor of a fixture safetensors file.
pub const Tensor = struct { name: []const u8, shape: []const u64, data: []const f32 };

/// The safetensors bytes of `tensors`, in order (8-byte header length, JSON header, data). Caller frees.
pub fn safetensors(a: std.mem.Allocator, tensors: []const Tensor) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.append(a, '{');
    var off: u64 = 0;
    for (tensors, 0..) |t, i| {
        const n = t.data.len * 4;
        try hdr.print(a, "{s}\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{ if (i == 0) "" else ",", t.name });
        for (t.shape, 0..) |d, k| try hdr.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + n });
        off += n;
    }
    try hdr.append(a, '}');
    const out = try a.alloc(u8, @intCast(8 + hdr.items.len + off));
    std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
    @memcpy(out[8..][0..hdr.items.len], hdr.items);
    var at: usize = 8 + hdr.items.len;
    for (tensors) |t| {
        @memcpy(out[at..][0 .. t.data.len * 4], std.mem.sliceAsBytes(t.data));
        at += t.data.len * 4;
    }
    return out;
}

/// `n` normal draws scaled by `scale`, seeded: a fixture's random weights. Caller frees.
pub fn randomF32(a: std.mem.Allocator, seed: u64, n: usize, scale: f32) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(f32, n);
    for (out) |*x| x.* = prng.random().floatNorm(f32) * scale;
    return out;
}

// ── tests ──

const testing = std.testing;

test "fixture: a memory pack reads its files within the bound and refuses past it" {
    const st = try safetensors(testing.allocator, &.{.{ .name = "w", .shape = &.{2}, .data = &.{ 1, 2 } }});
    defer testing.allocator.free(st);
    const mem: MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = "{\"a\":1}" }, .{ .name = "sub/model.safetensors", .bytes = st } } };
    const dir = mem.packDir();
    try testing.expect(dir.has("config.json") and dir.has("sub") and !dir.has("su") and !dir.has("nope"));
    var buf: [64]u8 = undefined;
    var why: Verdict = .{};
    try testing.expectEqualStrings("{\"a\":1}", dir.readFile("config.json", &buf, &why).?);
    try testing.expect(dir.readFile("config.json", buf[0..3], &why) == null and why.code == .too_large);
    try testing.expect(dir.readFile("missing.json", &buf, &why) == null and why.code == .failed);
    var len: usize = 0;
    try testing.expect(dir.safetensors_header(dir.ctx, Str.of("sub/model.safetensors"), &buf, buf.len, &len).isOk());
    try testing.expectEqualStrings("{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", buf[0..len]);
    try testing.expectEqual(@as(u64, st.len), dir.fileBytes("sub/model.safetensors"));
    try testing.expectEqual(@as(u64, 0), dir.fileBytes("sub"));
    var lbuf: [64]u8 = undefined;
    var it = dir.entries("", &lbuf).?;
    try testing.expectEqualStrings("config.json", it.next().?);
    try testing.expectEqualStrings("sub/", it.next().?);
    try testing.expect(it.next() == null);
    var sub = dir.entries("sub", &lbuf).?;
    try testing.expectEqualStrings("model.safetensors", sub.next().?);
    try testing.expect(sub.next() == null and dir.entries("nope", &lbuf) == null);
}

test "fixture: random weights are seeded" {
    const x = try randomF32(testing.allocator, 7, 4, 1);
    defer testing.allocator.free(x);
    const y = try randomF32(testing.allocator, 7, 4, 1);
    defer testing.allocator.free(y);
    try testing.expectEqualSlices(f32, x, y);
}
