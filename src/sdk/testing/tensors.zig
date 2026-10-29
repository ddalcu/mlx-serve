//! A fixture's safetensors file built tensor by tensor: names formatted in place, seeded random or constant data,
//! in F32, BF16, F16 or the integer dtypes a quantized pack stores (U16: EXL3 trellis codes). MLX-free; an arena
//! allocator is expected.

const std = @import("std");

pub const Dtype = enum {
    F32,
    BF16,
    F16,
    U32,
    I32,
    U16,

    fn width(d: Dtype) usize {
        return if (d == .BF16 or d == .F16 or d == .U16) 2 else 4;
    }
};

/// `alias` = the index of an earlier entry whose bytes this one shares (same offsets in the file).
const Entry = struct { name: []const u8, dtype: Dtype, shape: []const u64, bytes: []const u8, alias: ?usize = null };

pub const Tensors = struct {
    a: std.mem.Allocator,
    prng: std.Random.DefaultPrng,
    list: std.ArrayList(Entry) = .empty,

    pub fn init(a: std.mem.Allocator, seed: u64) Tensors {
        return .{ .a = a, .prng = .init(seed) };
    }

    fn count(shape: []const u64) usize {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        return n;
    }

    fn put(t: *Tensors, comptime fmt: []const u8, args: anytype, dtype: Dtype, shape: []const u64, data: []const u8) !void {
        try t.list.append(t.a, .{ .name = try std.fmt.allocPrint(t.a, fmt, args), .dtype = dtype, .shape = try t.a.dupe(u64, shape), .bytes = data });
    }

    /// Normal draws times `scale`, stored as `dtype` (F32, BF16 or F16).
    pub fn rand(t: *Tensors, comptime fmt: []const u8, args: anytype, shape: []const u64, dtype: Dtype, scale: f32) !void {
        const n = count(shape);
        const out = try t.a.alloc(u8, n * dtype.width());
        for (0..n) |i| writeFloat(out, dtype, i, t.prng.random().floatNorm(f32) * scale);
        try t.put(fmt, args, dtype, shape, out);
    }

    /// Every element `v`, stored as `dtype` (F32, BF16 or F16).
    pub fn fill(t: *Tensors, comptime fmt: []const u8, args: anytype, shape: []const u64, dtype: Dtype, v: f32) !void {
        const n = count(shape);
        const out = try t.a.alloc(u8, n * dtype.width());
        for (0..n) |i| writeFloat(out, dtype, i, v);
        try t.put(fmt, args, dtype, shape, out);
    }

    /// The given `values` (one per element), stored as `dtype` (F32 or BF16).
    pub fn values(t: *Tensors, comptime fmt: []const u8, args: anytype, shape: []const u64, dtype: Dtype, vs: []const f32) !void {
        std.debug.assert(vs.len == count(shape));
        const out = try t.a.alloc(u8, vs.len * dtype.width());
        for (vs, 0..) |v, i| writeFloat(out, dtype, i, v);
        try t.put(fmt, args, dtype, shape, out);
    }

    /// Uniform random words (packed quantized weights).
    pub fn words(t: *Tensors, comptime fmt: []const u8, args: anytype, shape: []const u64) !void {
        const n = count(shape);
        const out = try t.a.alloc(u8, n * 4);
        t.prng.random().bytes(out);
        try t.put(fmt, args, .U32, shape, out);
    }

    /// Uniform random halfwords (EXL3 trellis codes).
    pub fn halfwords(t: *Tensors, comptime fmt: []const u8, args: anytype, shape: []const u64) !void {
        const out = try t.a.alloc(u8, count(shape) * 2);
        t.prng.random().bytes(out);
        try t.put(fmt, args, .U16, shape, out);
    }

    /// The safetensors bytes (8-byte header length, JSON header, data), in insertion order.
    /// `name` as a second tensor over `existing`'s bytes (dtype and shape too): fixed-width layers share one buffer.
    pub fn alias(t: *Tensors, comptime fmt: []const u8, args: anytype, existing: []const u8) !void {
        for (t.list.items, 0..) |e, i| if (std.mem.eql(u8, e.name, existing)) {
            const target = e.alias orelse i;
            return t.list.append(t.a, .{ .name = try std.fmt.allocPrint(t.a, fmt, args), .dtype = e.dtype, .shape = e.shape, .bytes = e.bytes, .alias = target });
        };
        return error.NoSuchTensor;
    }

    pub fn bytes(t: *const Tensors) ![]u8 {
        var hdr: std.ArrayList(u8) = .empty;
        try hdr.append(t.a, '{');
        const starts = try t.a.alloc(u64, t.list.items.len);
        var off: u64 = 0;
        for (t.list.items, 0..) |e, i| {
            starts[i] = if (e.alias) |j| starts[j] else off;
            if (e.alias == null) off += e.bytes.len;
            try hdr.print(t.a, "{s}\"{s}\":{{\"dtype\":\"{t}\",\"shape\":[", .{ if (i == 0) "" else ",", e.name, e.dtype });
            for (e.shape, 0..) |d, k| try hdr.print(t.a, "{s}{d}", .{ if (k == 0) "" else ",", d });
            try hdr.print(t.a, "],\"data_offsets\":[{d},{d}]}}", .{ starts[i], starts[i] + e.bytes.len });
        }
        try hdr.append(t.a, '}');
        const out = try t.a.alloc(u8, @intCast(8 + hdr.items.len + off));
        std.mem.writeInt(u64, out[0..8], hdr.items.len, .little);
        @memcpy(out[8..][0..hdr.items.len], hdr.items);
        var at: usize = 8 + hdr.items.len;
        for (t.list.items) |e| if (e.alias == null) {
            @memcpy(out[at..][0..e.bytes.len], e.bytes);
            at += e.bytes.len;
        };
        return out;
    }
};

fn writeFloat(out: []u8, dtype: Dtype, i: usize, v: f32) void {
    switch (dtype) {
        .F32 => std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little),
        // Round to nearest even, as torch's bf16 cast does.
        .BF16 => {
            const b: u32 = @bitCast(v);
            std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16), .little);
        },
        .F16 => std.mem.writeInt(u16, out[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
        .U32, .I32, .U16 => unreachable,
    }
}

// ── tests ──

const testing = std.testing;

test "fixture tensors: a header names each tensor's dtype, shape and offsets in order" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t = Tensors.init(arena.allocator(), 1);
    try t.fill("a.{d}", .{0}, &.{2}, .F32, 1.5);
    try t.fill("b", .{}, &.{1}, .BF16, 1.0);
    const st = try t.bytes();
    const n = std.mem.readInt(u64, st[0..8], .little);
    try testing.expectEqualStrings(
        "{\"a.0\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[8,10]}}",
        st[8..][0..n],
    );
    try testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, st[8 + n + 8 ..][0..2], .little));
}

test "fixture tensors: an alias shares its target's offsets and adds no bytes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var t = Tensors.init(arena.allocator(), 1);
    try t.fill("a", .{}, &.{2}, .F32, 1.5);
    try t.fill("b", .{}, &.{1}, .F32, 2);
    try t.alias("c.{d}", .{0}, "a");
    try t.alias("d", .{}, "c.0");
    const st = try t.bytes();
    const n = std.mem.readInt(u64, st[0..8], .little);
    try testing.expectEqualStrings(
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}," ++
            "\"c.0\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},\"d\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}",
        st[8..][0..n],
    );
    try testing.expectEqual(8 + n + 12, st.len);
    try testing.expectError(error.NoSuchTensor, t.alias("e", .{}, "zz"));
}
