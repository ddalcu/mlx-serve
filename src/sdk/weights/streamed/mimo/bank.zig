//! Byte-preserving six-component MXFP4 expert records.
const std = @import("std");
const env = @import("mlx_host").env;
const io_util = @import("sdk").io_util;
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const Module = @This();

pub const format = "mlx-stream-mxfp4-expert-bank-v1";
pub const n_components = 6;
pub const gu_components = 4;
pub const routed_top_k = 8;
pub const Records = sdk_ext.expert.Records(n_components, gu_components);
pub const Dtype = enum { U32, U8 };
pub const Component = enum(u8) {
    gate_weight,
    gate_scales,
    up_weight,
    up_scales,
    down_weight,
    down_scales,
    pub fn name(c: Component) []const u8 {
        return names[@backingInt(c)];
    }
    const names = [_][]const u8{
        "gate_proj.weight", "gate_proj.scales", "up_proj.weight", "up_proj.scales", "down_proj.weight", "down_proj.scales",
    };
};
pub const Segment = struct { dtype: Dtype, rank: u8 = 2, shape: [2]u64, offset: u64, length: u64 };

pub fn mlxDtype(dtype: Dtype) mlx.mlx_dtype {
    return switch (dtype) {
        .U32 => .uint32,
        .U8 => .uint8,
    };
}

pub fn BankArraysOf(comptime T: type) type {
    return sdk_ext.quant.BankArrays(sdk_ext.quant.GatherQmm.Arrays(T));
}

pub const BankArrays = BankArraysOf(mlx.mlx_array);

pub fn bankArraysOf(arrays: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .w = arrays[0], .scales = arrays[1] },
        .up = .{ .w = arrays[2], .scales = arrays[3] },
        .down = .{ .w = arrays[4], .scales = arrays[5] },
    };
}

/// One SDK routed layer. The bytes and rank come from the validated manifest's Geometry.
pub const Layer = struct {
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};

fn layerOf(g: *const Geometry, routed_layer: u32) Layer {
    return .{
        .logical_bytes = g.record_bytes,
        .base_offset = @as(u64, routed_layer) * g.num_experts * g.record_bytes,
        .segments = g.segments,
    };
}

/// SDK view of an already-open Bank. Owns only its ranked layer table; the
/// original Bank remains the sole fd/digest owner and must outlive this view
/// and its Stream. Destroy the Stream, then this view, then the original Bank.
pub const SDKBank = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    sidecar: sdk_ext.expert.UncachedFd,

    pub fn init(a: std.mem.Allocator, owner: *const Bank) !SDKBank {
        if (owner.sidecar_fd < 0 or owner.sidecar_file_size != owner.geometry.total_bytes) return error.SidecarNotOpen;
        const g = &owner.geometry;
        const layers = try a.alloc(Layer, g.num_moe_layers);
        for (layers, 0..) |*layer, i| layer.* = layerOf(g, @intCast(i));
        return .{
            .allocator = a,
            .layers = layers,
            .n_experts = g.num_experts,
            .first_moe_layer = g.first_moe_layer,
            .sidecar = .{ .fd = owner.sidecar_fd, .size = owner.sidecar_file_size },
        };
    }

    /// The reader has already stopped. The borrowed fd stays open.
    pub fn deinit(b: *SDKBank) void {
        b.allocator.free(b.layers);
        b.* = undefined;
    }

    /// SDK index 0 is the manifest's first_moe_layer, not absolute model layer 0.
    pub fn recordOffset(b: *const SDKBank, layer: u32, expert: u32) u64 {
        const l = &b.layers[layer];
        return l.base_offset + @as(u64, expert) * l.logical_bytes;
    }

    pub fn spans(b: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }
};

/// StreamOf's bank module, with the borrowed SDKBank as its Bank type. The
/// owning Bank.parse/open/deinit API below remains the manifest/file API.
pub const SDK = struct {
    pub const n_components = Module.n_components;
    pub const gu_components = Module.gu_components;
    pub const Records = Module.Records;
    pub const Component = Module.Component;
    pub const Layer = Module.Layer;
    pub const Bank = SDKBank;
    pub const mlxDtype = Module.mlxDtype;
    pub const BankArrays = Module.BankArrays;
    pub const bankArraysOf = Module.bankArraysOf;
    pub const routed_top_k = Module.routed_top_k;
};

/// Pure record geometry: no MLX, file reads, or native reader initialization.
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]Segment,

    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, layers: u64) !Geometry {
        if (hidden == 0 or inter == 0 or hidden > 1 << 16 or inter > 1 << 16 or hidden % 32 != 0 or inter % 32 != 0 or
            experts == 0 or experts >= std.math.maxInt(u16) or layers == 0 or layers > 4096 or first > 4096 or first + layers > 4096 or experts * layers > 1 << 20)
            return error.InvalidGeometry;
        var g: Geometry = .{
            .hidden_size = @intCast(hidden),
            .intermediate_size = @intCast(inter),
            .num_experts = @intCast(experts),
            .first_moe_layer = @intCast(first),
            .num_moe_layers = @intCast(layers),
            .record_bytes = 0,
            .gate_up_bytes = 0,
            .total_bytes = 0,
            .segments = undefined,
        };
        var off: u64 = 0;
        for (0..3) |p| {
            const in = if (p == 2) inter else hidden;
            const out = if (p == 2) hidden else inter;
            g.segments[p * 2] = .{ .dtype = .U32, .shape = .{ out, in / 8 }, .offset = off, .length = out * in / 2 };
            off += g.segments[p * 2].length;
            g.segments[p * 2 + 1] = .{ .dtype = .U8, .shape = .{ out, in / 32 }, .offset = off, .length = out * in / 32 };
            off += g.segments[p * 2 + 1].length;
        }
        g.record_bytes = off;
        g.gate_up_bytes = g.segments[gu_components].offset;
        g.total_bytes = std.math.mul(u64, off, experts * layers) catch return error.InvalidGeometry;
        if (g.total_bytes > std.math.maxInt(i64)) return error.InvalidGeometry;
        return g;
    }

    /// Absolute model layer ID; its range was fixed at installation.
    pub fn recordOffset(g: *const Geometry, layer: u32, expert: u32) u64 {
        return (@as(u64, layer - g.first_moe_layer) * g.num_experts + expert) * g.record_bytes;
    }

    pub fn requireMiMo(g: *const Geometry) !void {
        if (g.hidden_size != 4096 or g.intermediate_size != 2048 or g.num_experts != 256 or g.first_moe_layer != 1 or g.num_moe_layers != 47)
            return error.UnsupportedTopology;
    }
};

const Manifest = struct {
    format: []const u8,
    data_file: []const u8,
    hidden_size: u64,
    intermediate_size: u64,
    num_experts: u64,
    first_moe_layer: u64,
    num_moe_layers: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    quantization: struct { mode: []const u8, bits: u32, group_size: u32 },
    source: struct { repository: []const u8, revision: []const u8 },
    segments: []const struct { name: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 },
    records: []const struct { layer: u64, expert: u64, offset: u64, length: u64, sha256: []const u8 },
    output_files: std.json.Value,
};

pub const Bank = struct {
    allocator: std.mem.Allocator,
    geometry: Geometry,
    digests: [][32]u8,
    source_repository: []const u8,
    source_revision: []const u8,
    sidecar_fd: std.c.fd_t = -1,
    sidecar_file_size: u64 = 0,

    /// Validate the published manifest, including every record's hash declaration.
    /// Payload verification belongs to the offline repacker/parity gate; opening a
    /// model never reads or hashes the entire 150-GiB bank.
    pub fn parse(a: std.mem.Allocator, text: []const u8) !Bank {
        const parsed = try std.json.parseFromSlice(Manifest, a, text, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const m = parsed.value;
        if (!std.mem.eql(u8, m.format, format) or !std.mem.eql(u8, m.data_file, "experts.bin")) return error.InvalidFormat;
        if (!std.mem.eql(u8, m.quantization.mode, "mxfp4") or m.quantization.bits != 4 or m.quantization.group_size != 32) return error.InvalidQuantization;
        const g = try Geometry.init(m.hidden_size, m.intermediate_size, m.num_experts, m.first_moe_layer, m.num_moe_layers);
        if (m.record_bytes != g.record_bytes or m.gate_up_bytes != g.gate_up_bytes) return error.InvalidGeometry;
        if (m.segments.len != n_components) return error.InvalidSegment;
        for (m.segments, g.segments, 0..) |got, want, i| {
            const component: Component = @fromBackingInt(@intCast(i));
            if (!std.mem.eql(u8, got.name, component.name()) or !std.mem.eql(u8, got.dtype, @tagName(want.dtype)) or
                !std.mem.eql(u64, got.shape, &want.shape) or got.offset != want.offset or got.length != want.length)
                return error.InvalidSegment;
        }
        if (m.source.repository.len == 0 or m.source.revision.len != 40) return error.InvalidSource;
        for (m.source.revision) |c| if (!std.ascii.isHex(c)) return error.InvalidSource;
        const count = @as(usize, g.num_experts) * g.num_moe_layers;
        if (m.records.len != count) return error.InvalidRecord;
        const digests = try a.alloc([32]u8, count);
        errdefer a.free(digests);
        for (m.records, digests, 0..) |r, *digest, i| {
            const layer = g.first_moe_layer + i / g.num_experts;
            const expert = i % g.num_experts;
            if (r.layer != layer or r.expert != expert or r.offset != i * g.record_bytes or r.length != g.record_bytes)
                return error.InvalidRecord;
            digest.* = try parseDigest(r.sha256);
        }
        if (m.output_files != .object) return error.InvalidOutputFile;
        const out = m.output_files.object.get("experts.bin") orelse return error.InvalidOutputFile;
        if (out != .object) return error.InvalidOutputFile;
        const size = out.object.get("size") orelse return error.InvalidOutputFile;
        const digest = out.object.get("sha256") orelse return error.InvalidOutputFile;
        if (size != .integer or size.integer < 0 or size.integer != g.total_bytes or digest != .string) return error.InvalidOutputFile;
        // The bank's digest is validated here (the reader re-hashes nothing per call).
        _ = try parseDigest(digest.string);
        const repository = try a.dupe(u8, m.source.repository);
        errdefer a.free(repository);
        const revision = try a.dupe(u8, m.source.revision);
        return .{
            .allocator = a,
            .geometry = g,
            .digests = digests,
            .source_repository = repository,
            .source_revision = revision,
        };
    }

    /// Construction only. Uses the existing F_NOCACHE reader contract; never
    /// follows an experts.bin symlink, and refuses a truncated/extended file.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
        var b = try readManifest(a, io, dir);
        errdefer b.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{dir}, 0);
        defer a.free(path);
        b.sidecar_fd = try io_util.openNoCache(path, .{ .follow_symlinks = false });
        var st: std.c.Stat = undefined;
        if (std.c.fstat(b.sidecar_fd, &st) != 0) return error.SidecarOpen;
        if (st.mode & std.c.S.IFMT != std.c.S.IFREG or st.size < 0 or @as(u64, @intCast(st.size)) != b.geometry.total_bytes)
            return error.SidecarGeometry;
        b.sidecar_file_size = @intCast(st.size);
        return b;
    }

    pub fn deinit(b: *Bank) void {
        if (b.sidecar_fd >= 0) _ = std.c.close(b.sidecar_fd);
        b.allocator.free(b.digests);
        b.allocator.free(b.source_repository);
        b.allocator.free(b.source_revision);
        b.* = undefined;
    }
};

fn parseDigest(hex: []const u8) ![32]u8 {
    if (hex.len != 64) return error.InvalidDigest;
    var digest: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&digest, hex) catch return error.InvalidDigest;
    return digest;
}

fn readManifest(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Bank {
    if (!std.fs.path.isAbsolute(dir)) return error.BankDirNotAbsolute;
    const path = try std.fs.path.join(a, &.{ dir, "expert-manifest.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(32 << 20));
    defer a.free(text);
    return Bank.parse(a, text);
}

/// Full manifest validation for admission/configuration inspection. Does not
/// open the expert sidecar, create MLX arrays, or start the process-global pool.
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    var bank = try readManifest(a, io, dir);
    defer bank.deinit();
    return bank.geometry;
}

/// A synthetic pack's sidecar for the stream and source tests: `bytes` written to a temporary `experts.bin`, opened past
/// the page cache, its owner `Bank` and the `SDKBank` over it. In place: the SDK bank keeps `&owner` (test code only).
pub const TestSidecar = struct {
    tmp: std.testing.TmpDir,
    fd: sdk_ext.expert.UncachedFd,
    owner: Bank,
    bank: SDKBank,

    pub fn init(f: *TestSidecar, a: std.mem.Allocator, geom: Geometry, bytes: []const u8) !void {
        f.tmp = std.testing.tmpDir(.{});
        errdefer f.tmp.cleanup();
        try f.tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts.bin", .data = bytes });
        var dir_buf: [512]u8 = undefined;
        var path_buf: [600]u8 = undefined;
        const dir = dir_buf[0..try f.tmp.dir.realPath(std.testing.io, &dir_buf)];
        const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/experts.bin", .{dir}, 0);
        f.fd = try sdk_ext.expert.openUncached(path, null);
        errdefer f.fd.close();
        f.owner = .{
            .allocator = a,
            .geometry = geom,
            .digests = &.{},
            .source_repository = "synthetic",
            .source_revision = "0000000000000000000000000000000000000000",
            .sidecar_fd = f.fd.fd,
            .sidecar_file_size = f.fd.size,
        };
        f.bank = try SDKBank.init(a, &f.owner);
    }

    /// After the streams over `bank` are gone.
    pub fn deinit(f: *TestSidecar) void {
        f.bank.deinit();
        f.fd.close();
        f.tmp.cleanup();
    }
};

test "mimo sdk2 mxfp4 bank: MiMo geometry preserves U32 weights and U8 scales" {
    const g = try Geometry.init(4096, 2048, 256, 1, 47);
    try std.testing.expectEqual(@as(u64, 13_369_344), g.record_bytes);
    try std.testing.expectEqual(@as(u64, 8_912_896), g.gate_up_bytes);
    try std.testing.expectEqual(@as(u64, 160_859_947_008), g.total_bytes);
    try std.testing.expectEqualSlices(u64, &.{ 2048, 512 }, &g.segments[0].shape);
    try std.testing.expectEqualSlices(u64, &.{ 4096, 64 }, &g.segments[5].shape);
    try std.testing.expectEqual(Dtype.U32, g.segments[0].dtype);
    try std.testing.expectEqual(Dtype.U8, g.segments[1].dtype);
    try std.testing.expectEqual(@as(u64, 256 * 13_369_344), g.recordOffset(2, 0));
}

test "mimo sdk2 mxfp4 bank: invalid geometry is refused before arithmetic" {
    try std.testing.expectError(error.InvalidGeometry, Geometry.init(33, 32, 2, 1, 1));
    try std.testing.expectError(error.InvalidGeometry, Geometry.init(32, 32, 0, 1, 1));
    try std.testing.expectError(error.InvalidGeometry, Geometry.init(32, 32, 2, 1, 0));
    try std.testing.expectError(error.InvalidGeometry, Geometry.init(1 << 31, 1 << 31, 256, 1, 47));
}

test "mimo sdk2 mxfp4 SDK bank: six ranked components and relative offsets preserve the manifest" {
    sdk_ext.expert.assertBank(SDK);
    try std.testing.expect(Records == sdk_ext.expert.Records(6, 4));
    const g = try Geometry.init(4096, 2048, 256, 1, 47);
    var layers = [_]Layer{ layerOf(&g, 0), layerOf(&g, 46) };
    const b: SDKBank = .{
        .allocator = std.testing.allocator,
        .layers = &layers,
        .n_experts = g.num_experts,
        .first_moe_layer = g.first_moe_layer,
        .sidecar = .{ .fd = -1, .size = g.total_bytes },
    };
    for (layers[0].segments, g.segments) |got, want| {
        try std.testing.expectEqual(@as(u8, 2), got.rank);
        try std.testing.expectEqual(want.offset, got.offset);
        try std.testing.expectEqual(want.length, got.length);
        try std.testing.expectEqualSlices(u64, &want.shape, &got.shape);
    }
    try std.testing.expectEqual(mlx.mlx_dtype.uint32, mlxDtype(layers[0].segments[0].dtype));
    try std.testing.expectEqual(mlx.mlx_dtype.uint8, mlxDtype(layers[0].segments[1].dtype));
    try std.testing.expectEqual(g.recordOffset(1, 7), b.recordOffset(0, 7));
    try std.testing.expectEqual(g.recordOffset(47, 255), b.recordOffset(1, 255));
    try std.testing.expectEqual(g.recordOffset(47, 255) + g.gate_up_bytes, b.spans(1, 255).down_offset);
    var owner = try Bank.parse(std.testing.allocator, fixture);
    defer owner.deinit();
    try std.testing.expectError(error.SidecarNotOpen, SDKBank.init(std.testing.allocator, &owner));
}

test "mimo sdk2 mxfp4 bank: manifest topology and per-record spans are construction invariants" {
    const text = fixture;
    var bank = try Bank.parse(std.testing.allocator, text);
    defer bank.deinit();
    try std.testing.expectEqual(@as(u64, 3264), bank.geometry.total_bytes);
    try std.testing.expectEqual(@as(usize, 2), bank.digests.len);
    const a = std.testing.allocator;
    const wrong_dtype = try std.mem.replaceOwned(u8, a, text, "\"U8\"", "\"F16\"");
    defer a.free(wrong_dtype);
    try std.testing.expectError(error.InvalidSegment, Bank.parse(a, wrong_dtype));
    const wrong_offset = try std.mem.replaceOwned(u8, a, text, "\"offset\":1632", "\"offset\":1633");
    defer a.free(wrong_offset);
    try std.testing.expectError(error.InvalidRecord, Bank.parse(a, wrong_offset));
    const wrong_mode = try std.mem.replaceOwned(u8, a, text, "\"mxfp4\"", "\"affine\"");
    defer a.free(wrong_mode);
    try std.testing.expectError(error.InvalidQuantization, Bank.parse(a, wrong_mode));
}

test "mimo sdk2 mxfp4 bank: inspect the Python repacker synthetic output without payload reads" {
    const path = env.get(.MIMO_Q4_BANK_FIXTURE) orelse return error.SkipZigTest;
    const g = try inspect(std.testing.allocator, std.testing.io, std.mem.span(path));
    try std.testing.expectEqual(@as(u32, 64), g.hidden_size);
    try std.testing.expectEqual(@as(u32, 32), g.intermediate_size);
    try std.testing.expectEqual(@as(u32, 2), g.num_experts);
    try std.testing.expectEqual(@as(u32, 1), g.first_moe_layer);
    try std.testing.expectEqual(@as(u64, 3264), g.record_bytes);
    try std.testing.expectEqual(@as(u64, 2176), g.gate_up_bytes);
}

const fixture =
    \\{"format":"mlx-stream-mxfp4-expert-bank-v1","data_file":"experts.bin",
    \\"hidden_size":32,"intermediate_size":32,"num_experts":2,"first_moe_layer":1,"num_moe_layers":1,
    \\"record_bytes":1632,"gate_up_bytes":1088,"quantization":{"mode":"mxfp4","bits":4,"group_size":32},
    \\"source":{"repository":"fixture","revision":"0123456789012345678901234567890123456789"},
    \\"segments":[
    \\{"name":"gate_proj.weight","dtype":"U32","shape":[32,4],"offset":0,"length":512},
    \\{"name":"gate_proj.scales","dtype":"U8","shape":[32,1],"offset":512,"length":32},
    \\{"name":"up_proj.weight","dtype":"U32","shape":[32,4],"offset":544,"length":512},
    \\{"name":"up_proj.scales","dtype":"U8","shape":[32,1],"offset":1056,"length":32},
    \\{"name":"down_proj.weight","dtype":"U32","shape":[32,4],"offset":1088,"length":512},
    \\{"name":"down_proj.scales","dtype":"U8","shape":[32,1],"offset":1600,"length":32}],
    \\"records":[
    \\{"layer":1,"expert":0,"offset":0,"length":1632,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    \\{"layer":1,"expert":1,"offset":1632,"length":1632,"sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],
    \\"output_files":{"experts.bin":{"size":3264,"sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}}}
;
