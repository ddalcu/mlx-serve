//! Bounded, allocation-free description of a V4.1 checkpoint's tensor inventory.
const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");

pub const max_header = 64 * 1024;
pub const max_tensors = 512;
const fixed_bytes = 48;
const spec_bytes = 56;
const magic = "DV41A001";

pub const Layout = struct {
    layers: usize,
    stages: usize,
    ring: usize,
    width: usize,
    index_width: usize,
    hidden_width: usize,
    dtype: mlx.mlx_dtype,
    kv_sources: u64,
    index_sources: u64,
    ratios: [64]u8 = @splat(0),
    engram: bool,
    settings: u64,

    pub fn source(self: Layout, i: usize) bool {
        return self.kv_sources & (@as(u64, 1) << @intCast(i)) != 0;
    }

    pub fn owner(self: Layout, i: usize) bool {
        return self.source(i) and self.index_sources & (@as(u64, 1) << @intCast(i)) != 0;
    }

    fn validate(self: Layout) !void {
        if (self.layers == 0 or self.layers > 64 or self.stages > 128) return error.Dsv41CacheGeometry;
        for ([_]usize{ self.ring, self.width, self.index_width }) |d| try dimension(d);
        if (self.stages > 0) try dimension(self.hidden_width);
        _ = try itemBytes(self.dtype);
        if (self.layers < 64 and (self.kv_sources | self.index_sources) >> @intCast(self.layers) != 0) return error.Dsv41CacheGeometry;
        for (0..self.layers) |i| {
            if (self.source(i) and self.ratios[i] == 0) return error.Dsv41CacheGeometry;
        }
    }

    fn key(self: Layout) u64 {
        var h = std.hash.Wyhash.init(0x41ca);
        std.hash.autoHash(&h, self);
        return h.final();
    }
};

pub const Kind = enum(u8) { win, comp, index, pend_kv, pend_gate, ds, mh, hist };

pub const Spec = struct {
    kind: Kind,
    layer: usize = 0,
    rows: usize,
    width: usize,
    dtype: mlx.mlx_dtype,

    pub fn rank(self: Spec) usize {
        return if (self.kind == .hist) 1 else 2;
    }

    pub fn name(self: Spec, buf: []u8) ![]const u8 {
        return switch (self.kind) {
            .ds => std.fmt.bufPrint(buf, "d{d}.win", .{self.layer}),
            .mh => "mh",
            .hist => "hist",
            else => std.fmt.bufPrint(buf, "l{d}.{s}", .{ self.layer, @tagName(self.kind) }),
        };
    }

    pub fn bytes(self: Spec) !u64 {
        try dimension(self.rows);
        try dimension(self.width);
        return checkedMul(try checkedMul(self.rows, self.width), try itemBytes(self.dtype));
    }

    fn words(self: Spec) ![7]u64 {
        return .{ @backingInt(self.kind), self.layer, self.rows, self.width, self.rank(), @intCast(@backingInt(self.dtype)), try self.bytes() };
    }
};

pub const Plan = struct {
    layout: Layout,
    n: usize,
    vision_key: u64,
    items: [max_tensors]Spec = undefined,
    len: usize = 0,
    loaded: u64 = 0,
    destination: u64 = 0,
    peak: u64 = 0,

    pub fn specs(self: *const Plan) []const Spec {
        return self.items[0..self.len];
    }

    pub fn init(l: Layout, n: usize, vision_key: u64) !Plan {
        try l.validate();
        try dimension(n);
        var p: Plan = .{ .layout = l, .n = n, .vision_key = vision_key };
        for (0..l.layers) |i| try p.layer(i);
        for (0..l.stages) |i| try p.add(.{ .kind = .ds, .layer = i, .rows = l.ring, .width = l.width, .dtype = l.dtype });
        if (l.stages > 0) try p.add(.{ .kind = .mh, .rows = 1, .width = l.hidden_width, .dtype = l.dtype });
        if (l.engram) try p.add(.{ .kind = .hist, .rows = n, .width = 1, .dtype = .uint32 });
        // Each copy/update is evaluated before the next. Allow a whole destination for non-donated
        // update scratch, and two input sets for file materialization and loader-copy overlap.
        const gpu = try checkedAdd(try checkedMul(p.loaded, 3), try checkedMul(p.destination, 2));
        // 4 KiB per tensor/state entry covers maps, descriptors and loaded/snapshot/restore wrappers.
        const host = try checkedAdd(try checkedMul(n, 12), try checkedAdd(2 * max_header, try checkedMul(p.len + l.layers + l.stages + 1, 4096)));
        p.peak = try checkedAdd(gpu, host);
        return p;
    }

    fn layer(self: *Plan, i: usize) !void {
        const l = self.layout;
        try self.add(.{ .kind = .win, .layer = i, .rows = l.ring, .width = l.width, .dtype = l.dtype });
        if (!l.source(i)) return;
        const used = self.n / l.ratios[i];
        try self.row(.comp, i, used, l.width);
        if (l.owner(i)) try self.row(.index, i, used, l.index_width);
        const pending = self.n % l.ratios[i];
        if (pending == 0) return;
        try self.add(.{ .kind = .pend_kv, .layer = i, .rows = pending, .width = l.width, .dtype = .float32 });
        try self.add(.{ .kind = .pend_gate, .layer = i, .rows = pending, .width = l.width, .dtype = .float32 });
    }

    fn row(self: *Plan, kind: Kind, i: usize, used: usize, width: usize) !void {
        const spec: Spec = .{ .kind = kind, .layer = i, .rows = used, .width = width, .dtype = self.layout.dtype };
        if (used > 0) try self.add(spec);
        const cap = try rowCapacity(used, try std.math.add(usize, used, 1));
        self.destination = try checkedAdd(self.destination, try checkedMul(try checkedMul(cap - used, width), try itemBytes(spec.dtype)));
    }

    fn add(self: *Plan, spec: Spec) !void {
        if (self.len == max_tensors) return error.Dsv41CacheGeometry;
        const bytes = try spec.bytes();
        self.loaded = try checkedAdd(self.loaded, bytes);
        self.destination = try checkedAdd(self.destination, bytes);
        self.items[self.len] = spec;
        self.len += 1;
    }

    pub fn encode(self: *const Plan, buf: []u8) ![]const u8 {
        const len = fixed_bytes + self.len * spec_bytes;
        if (len > max_header or len > buf.len) return error.Dsv41CacheStorage;
        @memcpy(buf[0..8], magic);
        const words = [_]u64{ self.layout.key(), self.n, self.vision_key, self.n, self.len };
        for (words, 0..) |w, i| put(buf, 8 + i * 8, w);
        for (self.specs(), 0..) |spec, i| {
            for (try spec.words(), 0..) |w, k| put(buf, fixed_bytes + i * spec_bytes + k * 8, w);
        }
        return buf[0..len];
    }

    pub fn decode(l: Layout, bytes: []const u8) !Plan {
        if (bytes.len < fixed_bytes or bytes.len > max_header) return error.Dsv41CacheHeader;
        const n = std.math.cast(usize, get(bytes, 16)) orelse return error.Dsv41CacheHeader;
        const p = Plan.init(l, n, get(bytes, 24)) catch return error.Dsv41CacheHeader;
        var buf: [max_header]u8 = undefined;
        const expected = try p.encode(&buf);
        if (!std.mem.eql(u8, expected, bytes)) return error.Dsv41CacheHeader;
        return p;
    }

    /// An unordered borrowed list is checked in full before the owner allocates any destination.
    pub fn validateArrays(self: *const Plan, arrays: []const abi.AuxTensor) ![max_tensors]usize {
        if (arrays.len != self.len) return error.Dsv41CacheInventory;
        var order: [max_tensors]usize = undefined;
        var seen: [max_tensors]bool = @splat(false);
        for (arrays, 0..) |tensor, i| {
            if (tensor.size < @sizeOf(abi.AuxTensor) or tensor.array.ctx == null) return error.Dsv41CacheTensor;
            const k = try self.find(tensor.name.slice());
            if (seen[k]) return error.Dsv41CacheDuplicate;
            seen[k] = true;
            try validateArray(self.items[k], .{ .ctx = tensor.array.ctx });
            order[k] = i;
        }
        return order;
    }

    fn find(self: *const Plan, name: []const u8) !usize {
        var buf: [40]u8 = undefined;
        for (self.specs(), 0..) |spec, i| {
            if (std.mem.eql(u8, try spec.name(&buf), name)) return i;
        }
        return error.Dsv41CacheTensorName;
    }
};

pub fn validateArray(spec: Spec, x: mlx.mlx_array) !void {
    if (x.ctx == null or mlx.mlx_array_dtype(x) != spec.dtype) return error.Dsv41CacheDtype;
    const sh = mlx.getShape(x);
    if (sh.len != spec.rank()) return error.Dsv41CacheShape;
    if (sh[0] != (std.math.cast(c_int, spec.rows) orelse return error.Dsv41CacheShape)) return error.Dsv41CacheShape;
    if (sh.len == 2 and sh[1] != (std.math.cast(c_int, spec.width) orelse return error.Dsv41CacheShape)) return error.Dsv41CacheShape;
    if ((mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)) != try spec.bytes()) return error.Dsv41CacheShape;
}

/// The ordinary quarter-growth rule, also applied to an imported exact used capacity for one more row.
pub fn rowCapacity(cap: usize, need: usize) !usize {
    const grown = std.math.add(usize, cap, cap / 4) catch return error.Dsv41CacheGeometry;
    const n = @max(need, @max(64, grown));
    try dimension(n);
    return n;
}

fn dimension(n: usize) !void {
    if (n == 0 or n > std.math.maxInt(c_int)) return error.Dsv41CacheGeometry;
}

fn itemBytes(dt: mlx.mlx_dtype) !u64 {
    return switch (dt) {
        .float16, .bfloat16 => 2,
        .float32, .uint32 => 4,
        else => error.Dsv41CacheGeometry,
    };
}

fn checkedAdd(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch return error.Dsv41CacheGeometry;
}

fn checkedMul(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch return error.Dsv41CacheGeometry;
}

fn put(buf: []u8, off: usize, n: u64) void {
    std.mem.writeInt(u64, buf[off..][0..8], n, .little);
}

fn get(buf: []const u8, off: usize) u64 {
    return std.mem.readInt(u64, buf[off..][0..8], .little);
}

// ── tests ──

const testing = std.testing;

fn testLayout() Layout {
    var l: Layout = .{ .layers = 3, .stages = 2, .ring = 24, .width = 64, .index_width = 32, .hidden_width = 256, .dtype = .bfloat16, .kv_sources = 6, .index_sources = 2, .engram = true, .settings = 123 };
    l.ratios[1] = 2;
    l.ratios[2] = 1;
    return l;
}

test "codec planning rejects malformed headers, geometry, settings and overflowing boundaries without MLX operations" {
    const l = testLayout();
    const p = try Plan.init(l, 9, 42);
    var buf: [max_header]u8 = undefined;
    const bytes = try p.encode(&buf);
    const ops = mlx.op_count.load(.monotonic);
    const decoded = try Plan.decode(l, bytes);
    try testing.expectEqual(ops, mlx.op_count.load(.monotonic));
    try testing.expectEqual(p.peak, decoded.peak);
    for (0..bytes.len) |len| try testing.expectError(error.Dsv41CacheHeader, Plan.decode(l, bytes[0..len]));
    // The media key varies by boundary; the remaining assertions match the loaded instance.
    for ([_]usize{ 0, 8, 16, 32, 40, 48, 56, 64, 72, 80, 88, 96 }) |off| {
        buf[off] ^= 0x80;
        try testing.expectError(error.Dsv41CacheHeader, Plan.decode(l, bytes));
        buf[off] ^= 0x80;
    }
    var changed = l;
    changed.dtype = .float16;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    changed = l;
    changed.settings += 1;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    changed = l;
    changed.kv_sources = 2;
    try testing.expectError(error.Dsv41CacheHeader, Plan.decode(changed, bytes));
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(l, 0, 0));
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(l, std.math.maxInt(usize), 0));
    changed = l;
    changed.width = std.math.maxInt(usize);
    try testing.expectError(error.Dsv41CacheGeometry, Plan.init(changed, 9, 0));
}

test "codec peak includes all inputs, padded destinations, scratch and host storage" {
    const p = try Plan.init(testLayout(), 9, 0);
    // Source buffers grow to 64 rows; rings, pending f32 rows, DSpark and history all coexist.
    const loaded: u64 = 3 * 24 * 64 * 2 + (4 * 64 + 9 * 64 + 4 * 32) * 2 + 2 * 64 * 4 + 2 * 24 * 64 * 2 + 256 * 2 + 9 * 4;
    const destination: u64 = loaded + ((64 - 4) * 64 + (64 - 9) * 64 + (64 - 4) * 32) * 2;
    try testing.expectEqual(loaded, p.loaded);
    try testing.expectEqual(destination, p.destination);
    const host: u64 = 3 * 9 * 4 + 2 * max_header + (12 + 3 + 2 + 1) * 4096;
    try testing.expect(p.peak >= 3 * loaded + 2 * destination + host);
    try testing.expectEqual(@as(usize, 64), try rowCapacity(0, 1));
    try testing.expectEqual(@as(usize, 80), try rowCapacity(64, 65));
    try testing.expectEqual(@as(usize, 100), try rowCapacity(80, 81));
    try testing.expectError(error.Dsv41CacheGeometry, rowCapacity(std.math.maxInt(usize), 1));
    try testing.expectError(error.Dsv41CacheGeometry, rowCapacity(0, @as(usize, std.math.maxInt(c_int)) + 1));
}
