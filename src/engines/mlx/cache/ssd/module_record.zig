//! Immutable module checkpoint envelopes and safetensors inventory validation. No MLX calls.

const std = @import("std");

pub const MAX_ENVELOPE_BYTES: usize = 64 * 1024;
pub const MAX_HEADER_BYTES: usize = 64 * 1024;
pub const PART_PAYLOAD_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_TOKEN_BYTES: u64 = 64 * 1024 * 1024;

pub const State = enum { registered, exporting, submitted, committed, failed, cancelled };

pub const Tensor = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: u64,
    part: u32 = 0,
};

pub const Media = struct { start: u32, key: i64 };

pub const Envelope = struct {
    v: u8 = 10,
    kind: []const u8 = "module",
    n: u32,
    prefix_key: i64,
    has_tools: bool,
    scheme: []const u8,
    bits: u8,
    group_size: u32,
    /// Loaded settings and the sorted pack stat stamp, supplied before scan.
    identity: []const u8,
    header_bytes: u64,
    token_bytes: u64,
    tensors: []const Tensor,
    media: []const Media = &.{},
    part_bytes: []const u64 = &.{},
    bytes: u64 = 0,
};

/// The binding and identity must outlive the tier. Validation is allocation-free and must
/// compare the codec header AND this inventory with the loaded model's geometry/settings.
pub const Context = struct {
    identity: []const u8,
    user: ?*anyopaque = null,
    validate: *const fn (?*anyopaque, []const u8, *const Envelope) bool,
};

pub const TokenMap = union(enum) { absent, builtin, file: []const u8 };
const PackFile = struct { name: []const u8, present: bool, bytes: u64 = 0, mtime: i128 = 0 };

/// Names come from the loaded pack's weight selection, including applicable auxiliary weights.
/// Only stats enter the identity; token-map absence and the embedded fallback are distinct.
pub fn packStamp(a: std.mem.Allocator, io: std.Io, root: []const u8, files: []const []const u8, token_map: TokenMap) ![]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, root, .{});
    defer dir.close(io);
    const names = try a.dupe([]const u8, files);
    defer a.free(names);
    std.mem.sort([]const u8, names, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.less);
    const stamps = try a.alloc(PackFile, names.len);
    defer a.free(stamps);
    for (names, 0..) |name, i| {
        if (i > 0 and std.mem.eql(u8, names[i - 1], name)) return error.BadPackStamp;
        stamps[i] = try statPackFile(io, dir, name);
    }
    const map_file: ?PackFile = switch (token_map) {
        .file => |name| try statPackFile(io, dir, name),
        else => null,
    };
    return std.json.Stringify.valueAlloc(a, .{ .files = stamps, .token_map = @tagName(token_map), .token_map_file = map_file }, .{});
}

fn statPackFile(io: std.Io, dir: std.Io.Dir, name: []const u8) !PackFile {
    if (name.len == 0 or std.fs.path.isAbsolute(name) or std.mem.indexOfScalar(u8, name, '\\') != null) return error.BadPackStamp;
    var parts = std.mem.splitScalar(u8, name, '/');
    while (parts.next()) |part| {
        if (std.mem.eql(u8, part, "..") or std.mem.eql(u8, part, ".") or part.len == 0) return error.BadPackStamp;
    }
    const stat = dir.statFile(io, name, .{}) catch |err| {
        if (err == error.FileNotFound) return .{ .name = name, .present = false };
        return err;
    };
    if (stat.kind != .file) return error.BadPackStamp;
    return .{ .name = name, .present = true, .bytes = stat.size, .mtime = stat.mtime.nanoseconds };
}

pub const Record = struct {
    arena: std.heap.ArenaAllocator,
    manifest: Envelope,
    header: []const u8,
    meta: []const u8,
    directory: []const u8 = &.{},
    state: State = .registered,
    /// Files accepted by Writer, including those still queued or in flight.
    accepted_bytes: u64 = 0,
    cursor: usize = 0,
    loss_counted: bool = false,

    pub fn create(a: std.mem.Allocator, input: Envelope, header: []const u8) !Record {
        try validateEnvelope(input, header);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const aa = arena.allocator();
        const initial = try std.json.Stringify.valueAlloc(aa, input, .{});
        if (initial.len > MAX_ENVELOPE_BYTES) return error.ModuleEnvelopeTooLarge;
        var m = try std.json.parseFromSliceLeaky(Envelope, aa, initial, .{ .allocate = .alloc_always });
        const tensors = try aa.dupe(Tensor, m.tensors);
        const sizes = try planParts(aa, tensors);
        m.tensors = tensors;
        m.part_bytes = sizes;
        const meta = try encodeEnvelope(aa, &m);
        const owned_header = try aa.dupe(u8, header);
        return .{ .arena = arena, .manifest = m, .header = owned_header, .meta = meta };
    }

    pub fn parse(a: std.mem.Allocator, meta: []const u8, header: []const u8) !Record {
        if (meta.len > MAX_ENVELOPE_BYTES) return error.ModuleEnvelopeTooLarge;
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const aa = arena.allocator();
        const Tag = struct { v: u8, kind: []const u8 };
        const tag = try std.json.parseFromSliceLeaky(Tag, aa, meta, .{ .ignore_unknown_fields = true });
        if (tag.v != 10 or !std.mem.eql(u8, tag.kind, "module")) return error.BadModuleInventory;
        const m = try std.json.parseFromSliceLeaky(Envelope, aa, meta, .{ .allocate = .alloc_always });
        try validateEnvelope(m, header);
        const planned = try aa.dupe(Tensor, m.tensors);
        const sizes = try planParts(aa, planned);
        if (!std.mem.eql(u64, sizes, m.part_bytes)) return error.BadModuleInventory;
        for (planned, m.tensors) |want, have| if (want.part != have.part) return error.BadModuleInventory;
        if (try footprint(m, meta.len) != m.bytes) return error.BadModuleInventory;
        const owned_header = try aa.dupe(u8, header);
        const owned_meta = try aa.dupe(u8, meta);
        return .{
            .arena = arena,
            .manifest = m,
            .header = owned_header,
            .meta = owned_meta,
            .state = .committed,
            .accepted_bytes = m.bytes,
        };
    }

    pub fn deinit(self: *Record) void {
        self.arena.deinit();
    }
};

fn validateEnvelope(m: Envelope, header: []const u8) !void {
    if (m.v != 10 or !std.mem.eql(u8, m.kind, "module")) return error.BadModuleInventory;
    if (header.len == 0 or header.len > MAX_HEADER_BYTES or m.header_bytes != header.len) return error.BadModuleInventory;
    if (m.n == 0 or m.token_bytes != @as(u64, m.n) * 4 or m.token_bytes > MAX_TOKEN_BYTES) return error.BadModuleInventory;
    if (m.identity.len == 0 or m.tensors.len == 0) return error.BadModuleInventory;
    try validateTensors(m.tensors);
    var previous: ?u32 = null;
    for (m.media) |sp| {
        if (sp.start >= m.n) return error.BadModuleInventory;
        if (previous) |p| if (sp.start <= p) return error.BadModuleInventory;
        previous = sp.start;
    }
}

fn validateTensors(tensors: []const Tensor) !void {
    for (tensors, 0..) |t, i| {
        if (!canonicalName(t.name) or t.shape.len > 8) return error.BadModuleInventory;
        for (tensors[0..i]) |prior| if (std.mem.eql(u8, prior.name, t.name)) return error.BadModuleInventory;
        var bytes = dtypeBytes(t.dtype) orelse return error.BadModuleInventory;
        for (t.shape) |d| {
            if (d == 0 or d > std.math.maxInt(c_int)) return error.BadModuleInventory;
            bytes = std.math.mul(u64, bytes, d) catch return error.BadModuleInventory;
        }
        if (bytes != t.bytes) return error.BadModuleInventory;
    }
}

fn canonicalName(name: []const u8) bool {
    if (name.len == 0 or name.len > 256 or std.mem.eql(u8, name, "__metadata__")) return false;
    for (name) |c| {
        if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '.' and c != '-') return false;
    }
    return true;
}

fn dtypeBytes(dtype: []const u8) ?u64 {
    const names = [_][]const u8{ "BOOL", "U8", "I8", "U16", "I16", "F16", "BF16", "U32", "I32", "F32", "U64", "I64" };
    const widths = [_]u64{ 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 8, 8 };
    for (names, widths) |name, width| if (std.mem.eql(u8, name, dtype)) return width;
    return null;
}

fn planParts(a: std.mem.Allocator, tensors: []Tensor) ![]u64 {
    var part: u32 = 0;
    var payload: u64 = 0;
    for (tensors) |*t| {
        if (payload > 0 and t.bytes > PART_PAYLOAD_BYTES -| payload) {
            part += 1;
            payload = 0;
        }
        t.part = part;
        payload = try std.math.add(u64, payload, t.bytes);
    }
    const sizes = try a.alloc(u64, @as(usize, part) + 1);
    for (sizes, 0..) |*size, i| {
        const header = try partHeader(a, tensors, @intCast(i));
        defer a.free(header);
        var bytes: u64 = 8 + @as(u64, header.len);
        for (tensors) |t| {
            if (t.part == i) bytes = try std.math.add(u64, bytes, t.bytes);
        }
        size.* = bytes;
    }
    return sizes;
}

pub fn partHeader(a: std.mem.Allocator, tensors: []const Tensor, part: u32) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "{\"__metadata__\":{}");
    var offset: u64 = 0;
    for (tensors) |t| {
        if (t.part != part) continue;
        try out.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, i| {
            if (i > 0) try out.appendSlice(a, ",");
            try out.print(a, "{d}", .{d});
        }
        const end = try std.math.add(u64, offset, t.bytes);
        try out.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, end });
        offset = end;
    }
    try out.appendSlice(a, "}");
    if (out.items.len > MAX_ENVELOPE_BYTES) return error.ModuleEnvelopeTooLarge;
    return out.toOwnedSlice(a);
}

fn footprint(m: Envelope, meta_len: usize) !u64 {
    var bytes = try std.math.add(u64, m.token_bytes, m.header_bytes);
    bytes = try std.math.add(u64, bytes, meta_len);
    for (m.part_bytes) |size| bytes = try std.math.add(u64, bytes, size);
    return bytes;
}

fn encodeEnvelope(a: std.mem.Allocator, m: *Envelope) ![]u8 {
    for (0..16) |_| {
        const encoded = try std.json.Stringify.valueAlloc(a, m.*, .{});
        if (encoded.len > MAX_ENVELOPE_BYTES) return error.ModuleEnvelopeTooLarge;
        const bytes = try footprint(m.*, encoded.len);
        if (bytes == m.bytes) return encoded;
        a.free(encoded);
        m.bytes = bytes;
    }
    return error.BadModuleInventory;
}

pub fn validatePartHeader(a: std.mem.Allocator, header: []const u8, payload: u64, tensors: []const Tensor, part: u32) !void {
    if (header.len > MAX_ENVELOPE_BYTES) return error.BadModulePart;
    var parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.BadModulePart;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadModulePart;
    const obj = parsed.value.object;
    var count: usize = 0;
    var offset: u64 = 0;
    for (tensors) |t| {
        if (t.part != part) continue;
        try validateDescriptor(obj.get(t.name) orelse return error.BadModulePart, t, offset);
        offset = std.math.add(u64, offset, t.bytes) catch return error.BadModulePart;
        count += 1;
    }
    if (obj.get("__metadata__")) |meta| {
        if (meta != .object or meta.object.count() != 0) return error.BadModulePart;
        count += 1;
    }
    if (obj.count() != count or offset != payload) return error.BadModulePart;
}

fn unsigned(value: std.json.Value) !u64 {
    if (value != .integer or value.integer < 0) return error.BadModulePart;
    return @intCast(value.integer);
}

fn validateDescriptor(value: std.json.Value, t: Tensor, offset: u64) !void {
    if (value != .object or value.object.count() != 3) return error.BadModulePart;
    const obj = value.object;
    const dtype = obj.get("dtype") orelse return error.BadModulePart;
    if (dtype != .string or !std.mem.eql(u8, dtype.string, t.dtype)) return error.BadModulePart;
    const shape = obj.get("shape") orelse return error.BadModulePart;
    if (shape != .array or shape.array.items.len != t.shape.len) return error.BadModulePart;
    for (shape.array.items, t.shape) |d, want| if (try unsigned(d) != want) return error.BadModulePart;
    const offsets = obj.get("data_offsets") orelse return error.BadModulePart;
    if (offsets != .array or offsets.array.items.len != 2) return error.BadModulePart;
    if (try unsigned(offsets.array.items[0]) != offset) return error.BadModulePart;
    if (try unsigned(offsets.array.items[1]) != offset + t.bytes) return error.BadModulePart;
}

// ── Tests ──

const testing = std.testing;

test "module envelopes round-trip an exact encoded footprint" {
    var record = try Record.create(testing.allocator, .{
        .n = 512,
        .prefix_key = @bitCast(@as(u64, 0xfedcba9876543210)),
        .has_tools = true,
        .scheme = "off",
        .bits = 0,
        .group_size = 64,
        .identity = "weights:16:42;token-map:builtin",
        .header_bytes = 5,
        .token_bytes = 2048,
        .tensors = &.{.{ .name = "ring", .dtype = "F32", .shape = &.{4}, .bytes = 16 }},
    }, "codec");
    defer record.deinit();
    const m = record.manifest;
    try testing.expectEqual(@as(u8, 10), m.v);
    try testing.expectEqualStrings("module", m.kind);
    try testing.expectEqual(@as(u64, 2048 + 5 + record.meta.len + m.part_bytes[0]), m.bytes);
    var reopened = try Record.parse(testing.allocator, record.meta, "codec");
    defer reopened.deinit();
    try testing.expectEqual(m.prefix_key, reopened.manifest.prefix_key);
    try testing.expectEqualSlices(u64, m.part_bytes, reopened.manifest.part_bytes);
}

test "module inventory rejects duplicate names and invalid dimensions" {
    const cases = [_][]const Tensor{
        &.{ .{ .name = "ring", .dtype = "F32", .shape = &.{4}, .bytes = 16 }, .{ .name = "ring", .dtype = "F32", .shape = &.{4}, .bytes = 16 } },
        &.{.{ .name = "ring", .dtype = "F32", .shape = &.{0}, .bytes = 0 }},
        &.{.{ .name = "ring", .dtype = "F32", .shape = &.{std.math.maxInt(u64)}, .bytes = 16 }},
        &.{.{ .name = "../ring", .dtype = "F32", .shape = &.{4}, .bytes = 16 }},
        &.{.{ .name = "ring", .dtype = "F32", .shape = &.{4}, .bytes = 8 }},
    };
    for (cases) |tensors| {
        try testing.expectError(error.BadModuleInventory, Record.create(testing.allocator, .{
            .n = 512, .prefix_key = 0, .has_tools = false, .scheme = "off", .bits = 0,
            .group_size = 64, .identity = "pack", .header_bytes = 1, .token_bytes = 2048, .tensors = tensors,
        }, "h"));
    }
}

test "module safetensors validation rejects duplicate keys and bad offsets" {
    const tensors: []const Tensor = &.{.{ .name = "ring", .dtype = "F32", .shape = &.{4}, .bytes = 16 }};
    const good = "{\"ring\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
    try validatePartHeader(testing.allocator, good, 16, tensors, 0);
    const bad = [_][]const u8{
        "{\"ring\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]},\"ring\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}",
        "{\"ring\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[1,17]}}",
        "{\"ring\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[0,16]}}",
        "{\"ring\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]}}",
        "{\"other\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}",
    };
    for (bad) |header| try testing.expectError(error.BadModulePart, validatePartHeader(testing.allocator, header, 16, tensors, 0));
}

test "pack stat stamps sort loaded files and distinguish token map fallback" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.safetensors", .data = "bb" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.safetensors", .data = "a" });
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const first = try packStamp(testing.allocator, testing.io, root, &.{ "b.safetensors", "a.safetensors", "dspark.safetensors" }, .builtin);
    defer testing.allocator.free(first);
    const reordered = try packStamp(testing.allocator, testing.io, root, &.{ "dspark.safetensors", "a.safetensors", "b.safetensors" }, .builtin);
    defer testing.allocator.free(reordered);
    try testing.expectEqualStrings(first, reordered);
    const absent = try packStamp(testing.allocator, testing.io, root, &.{ "b.safetensors", "a.safetensors", "dspark.safetensors" }, .absent);
    defer testing.allocator.free(absent);
    try testing.expect(!std.mem.eql(u8, first, absent));
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.safetensors", .data = "changed" });
    const changed = try packStamp(testing.allocator, testing.io, root, &.{ "b.safetensors", "a.safetensors", "dspark.safetensors" }, .builtin);
    defer testing.allocator.free(changed);
    try testing.expect(!std.mem.eql(u8, first, changed));
    const missing_map = try packStamp(testing.allocator, testing.io, root, &.{ "a.safetensors", "b.safetensors" }, .{ .file = "token_map.bin" });
    defer testing.allocator.free(missing_map);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "token_map.bin", .data = "map" });
    const selected_map = try packStamp(testing.allocator, testing.io, root, &.{ "a.safetensors", "b.safetensors" }, .{ .file = "token_map.bin" });
    defer testing.allocator.free(selected_map);
    try testing.expect(!std.mem.eql(u8, missing_map, selected_map));
}
