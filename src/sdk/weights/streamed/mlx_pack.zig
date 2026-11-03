//! A standard MLX pack's routed experts streamed as downloaded (mlx-lm / mlx-community layout, affine): every tensor
//! named `<prefix>.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` (experts stacked on axis 0) or
//! `<prefix>.experts.<e>.{gate,up,down}_proj.*` (one tensor per expert) is located in its shard through the
//! safetensors headers and read part by part into the expert stream's slot rows (`scatter`: an expert's nine parts live
//! in separate tensors, possibly separate shards). The MTP head's experts are not routed layers here. The stream, its
//! residency policy and the read pool are `sdk_ext.expert`'s; the MoE math stays the trunk's.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const expert = @import("sdk_ext.zig").expert;
const io = expert.io;
const Module = @This();

pub const n_components = 9;
pub const gu_components = 6;
pub const Records = io.Records(n_components, gu_components);
pub const scatter = true;
/// mlx-lm's mixed recipes give a pack at most two expert geometries (a layer at the higher width).
pub const per_geometry_transients = true;
/// Read by the lookahead selector only, which a scatter bank never installs.
pub const routed_top_k = 8;

pub const Component = enum(u8) { gate_weight, gate_scales, gate_biases, up_weight, up_scales, up_biases, down_weight, down_scales, down_biases };
pub const Dtype = enum { U32, BF16 };
pub const Segment = struct { dtype: Dtype, rank: u8 = 2, shape: [2]u64, offset: u64, length: u64 };

pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .U32 => .uint32,
        .BF16 => .bfloat16,
    };
}

/// One routed layer: its slot rows' segments (an expert's nine parts back to back in a virtual record) and, per expert,
/// where each part lies in the pack.
pub const Layer = struct {
    segments: [n_components]Segment,
    logical_bytes: u64,
    parts: [][n_components]Records.Part,
    prefix: []const u8,
};

pub const Proj = struct { w: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array };
pub const BankArrays = struct { gate: Proj, up: Proj, down: Proj };

pub fn bankArraysOf(a: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .w = a[0], .scales = a[1], .biases = a[2] },
        .up = .{ .w = a[3], .scales = a[4], .biases = a[5] },
        .down = .{ .w = a[6], .scales = a[7], .biases = a[8] },
    };
}

pub const S = expert.stream.StreamOf(Module, false);

/// The pack's routed layers in layer order, its shards opened past the page cache.
pub const Bank = struct {
    arena: std.heap.ArenaAllocator,
    layers: []Layer,
    n_experts: u32,
    /// The first shard: unread (a scatter bank reads part by part).
    sidecar: io.UncachedFd,
    fds: []io.UncachedFd,
    /// The routed experts' bytes in the pack.
    expert_bytes: u64,

    pub fn open(a: std.mem.Allocator, dir: []const u8) !Bank {
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const ar = arena.allocator();
        const names = try shardNames(ar, dir);
        var fds: std.ArrayList(io.UncachedFd) = .empty;
        errdefer for (fds.items) |f| f.close();
        var found: std.StringArrayHashMapUnmanaged(Found) = .empty;
        var expert_bytes: u64 = 0;
        for (names) |name| {
            const path = try std.fmt.allocPrintSentinel(ar, "{s}/{s}", .{ dir, name }, 0);
            const fd = io.openUncachedFollowing(path, null) catch return error.StreamedPackShard;
            try fds.append(ar, fd);
            expert_bytes += try scanShard(ar, fd, @intCast(fds.items.len - 1), &found);
        }
        if (found.count() == 0) return error.StreamedPackNoExperts;
        const layers = try ar.alloc(Layer, found.count());
        var n_experts: u32 = 0;
        for (found.keys(), found.values(), layers) |prefix, *f, *l| {
            l.* = try f.layer(ar, prefix, fds.items);
            if (n_experts == 0) n_experts = @intCast(l.parts.len);
            if (l.parts.len != n_experts) return error.StreamedPackExperts;
        }
        std.sort.pdq(Layer, layers, {}, struct {
            fn lt(_: void, x: Layer, y: Layer) bool {
                const lx = layerNumber(x.prefix);
                const ly = layerNumber(y.prefix);
                return if (lx != ly) lx < ly else std.mem.lessThan(u8, x.prefix, y.prefix);
            }
        }.lt);
        return .{ .arena = arena, .layers = layers, .n_experts = n_experts, .sidecar = fds.items[0], .fds = fds.items, .expert_bytes = expert_bytes };
    }

    /// After the stream over the bank is gone.
    pub fn deinit(b: *Bank) void {
        for (b.fds) |f| f.close();
        b.arena.deinit();
    }

    /// The routed layer whose tensors sit under `prefix` (e.g. `language_model.model.layers.3.mlp`).
    pub fn layerOf(b: *const Bank, prefix: []const u8) ?u32 {
        for (b.layers, 0..) |l, i| if (std.mem.eql(u8, l.prefix, prefix)) return @intCast(i);
        return null;
    }

    /// A virtual record order (layer, expert): the stream cuts its reads into parts by it.
    pub fn recordOffset(b: *const Bank, layer: u32, e: u32) u64 {
        return (@as(u64, layer) * b.n_experts + e) * b.layers[layer].logical_bytes;
    }

    pub fn spans(b: *const Bank, layer: u32, e: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, e);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }

    pub fn parts(b: *const Bank, layer: u32, e: u32) [n_components]Records.Part {
        return b.layers[layer].parts[e];
    }

    /// The slot rows a stream over the bank holds with `rows` per layer: every layer's rows and the transient scratch
    /// at each geometry.
    pub fn slotBytes(b: *const Bank, rows: u32) u64 {
        var n: u64 = 0;
        var geoms: [2]?*const Layer = .{ null, null };
        for (b.layers) |*l| {
            n += @as(u64, rows) * l.logical_bytes;
            for (&geoms) |*g| {
                if (g.*) |x| if (S.sameGeometry(x, l)) break;
                if (g.* == null) {
                    g.* = l;
                    break;
                }
            }
        }
        for (geoms) |g| if (g) |x| {
            n += @as(u64, S.max_route_ids) * x.logical_bytes;
        };
        return n;
    }
};

/// One tensor's bytes in a shard.
const Tensor = struct { file: u32, start: u64, len: u64, dtype: []const u8, shape: []const u64 };

/// A prefix's routed tensors as found: the stacked nine, or each expert's.
const Found = struct {
    stacked: [n_components]?Tensor = @splat(null),
    per: std.AutoArrayHashMapUnmanaged(u64, Tensor) = .empty,

    fn layer(f: *const Found, a: std.mem.Allocator, prefix: []const u8, fds: []const io.UncachedFd) !Layer {
        const stacked = f.per.count() == 0;
        if (!stacked) for (f.stacked) |t| if (t != null) return error.StreamedPackLayout;
        const n_experts: usize = if (stacked) blk: {
            const t = f.stacked[0] orelse return error.StreamedPackNotAffine;
            if (t.shape.len != 3) return error.StreamedPackShape;
            break :blk @intCast(t.shape[0]);
        } else std.mem.max(u64, f.per.keys()) / n_components + 1;
        if (n_experts == 0 or n_experts >= std.math.maxInt(u16)) return error.StreamedPackExperts;
        if (!stacked and f.per.count() != n_experts * n_components) return error.StreamedPackNotAffine;
        var l: Layer = .{ .segments = undefined, .logical_bytes = 0, .parts = try a.alloc([n_components]Records.Part, n_experts), .prefix = prefix };
        for (0..n_components) |c| {
            const elem: u64 = if (c % 3 == 0) 4 else 2;
            const want_dtype = if (c % 3 == 0) "U32" else "BF16";
            for (0..n_experts) |e| {
                const t = (if (stacked) f.stacked[c] else f.per.get(e * n_components + c)) orelse return error.StreamedPackNotAffine;
                if (!std.mem.eql(u8, t.dtype, want_dtype)) return error.StreamedPackDtype;
                const sh = if (stacked) t.shape[1..] else t.shape;
                if (sh.len != 2 or (stacked and (t.shape.len != 3 or t.shape[0] != n_experts))) return error.StreamedPackShape;
                const one = sh[0] * sh[1] * elem;
                if (t.len != one * @as(u64, if (stacked) n_experts else 1)) return error.StreamedPackShape;
                if (e == 0) {
                    l.segments[c] = .{ .dtype = if (c % 3 == 0) .U32 else .BF16, .shape = .{ sh[0], sh[1] }, .offset = l.logical_bytes, .length = one };
                    l.logical_bytes += one;
                } else if (!std.mem.eql(u64, &l.segments[c].shape, sh)) return error.StreamedPackShape;
                l.parts[e][c] = .{ .fd = fds[t.file], .offset = t.start + if (stacked) e * one else 0 };
            }
        }
        return l;
    }
};

/// The shards the index names (a single `model.safetensors` without one).
fn shardNames(a: std.mem.Allocator, dir: []const u8) ![]const []const u8 {
    const sio = std.Io.Threaded.global_single_threaded.io();
    var d = std.Io.Dir.openDirAbsolute(sio, dir, .{}) catch return error.StreamedPackShard;
    defer d.close(sio);
    const raw = d.readFileAlloc(sio, "model.safetensors.index.json", a, .limited(64 << 20)) catch
        return a.dupe([]const u8, &.{"model.safetensors"});
    const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return error.StreamedPackIndex;
    const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return error.StreamedPackIndex;
    if (wm != .object) return error.StreamedPackIndex;
    var set: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| {
        if (v != .string or v.string.len == 0 or v.string[0] == '/' or std.mem.indexOf(u8, v.string, "..") != null) return error.StreamedPackIndex;
        try set.put(a, v.string, {});
    }
    return set.keys();
}

/// Records the routed tensors of the shard behind `fd` in `found`; returns their bytes. A header that does not parse,
/// or a tensor past the end of the file, is refused by name.
fn scanShard(a: std.mem.Allocator, fd: io.UncachedFd, file: u32, found: *std.StringArrayHashMapUnmanaged(Found)) !u64 {
    var lenb: [8]u8 = undefined;
    if (fd.size < 8 or std.c.pread(fd.fd, &lenb, 8, 0) != 8) return error.StreamedPackTruncated;
    const hlen = std.mem.readInt(u64, &lenb, .little);
    if (hlen > fd.size - 8 or hlen > 256 << 20) return error.StreamedPackTruncated;
    const hbuf = try a.alloc(u8, @intCast(hlen));
    if (std.c.pread(fd.fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.StreamedPackTruncated;
    const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{}) catch return error.StreamedPackHeader;
    if (hdr != .object) return error.StreamedPackHeader;
    const base = 8 + hlen;
    var bytes: u64 = 0;
    var it = hdr.object.iterator();
    while (it.next()) |kv| {
        const r = routedOf(kv.key_ptr.*) orelse continue;
        const v = kv.value_ptr.*;
        if (v != .object) return error.StreamedPackHeader;
        const dt = v.object.get("dtype") orelse return error.StreamedPackHeader;
        const shape = v.object.get("shape") orelse return error.StreamedPackHeader;
        const offs = v.object.get("data_offsets") orelse return error.StreamedPackHeader;
        if (dt != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.StreamedPackHeader;
        const dims = try a.alloc(u64, shape.array.items.len);
        for (shape.array.items, dims) |x, *d| d.* = if (x == .integer and x.integer >= 0) @intCast(x.integer) else return error.StreamedPackHeader;
        const o = offs.array.items;
        if (o[0] != .integer or o[1] != .integer or o[0].integer < 0 or o[1].integer < o[0].integer) return error.StreamedPackHeader;
        const start = base + @as(u64, @intCast(o[0].integer));
        const len: u64 = @intCast(o[1].integer - o[0].integer);
        if (start + len > fd.size) return error.StreamedPackTruncated;
        const t: Tensor = .{ .file = file, .start = start, .len = len, .dtype = dt.string, .shape = dims };
        const gop = try found.getOrPut(a, r.prefix);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        if (r.expert) |e| {
            if (try gop.value_ptr.per.fetchPut(a, @as(u64, e) * n_components + r.comp, t) != null) return error.StreamedPackLayout;
        } else gop.value_ptr.stacked[r.comp] = t;
        bytes += len;
    }
    return bytes;
}

/// A routed expert tensor's layer prefix, expert (null: stacked) and component; null for any other tensor (the MTP
/// head's included).
fn routedOf(key: []const u8) ?struct { prefix: []const u8, expert: ?u32, comp: u8 } {
    if (std.mem.startsWith(u8, key, "mtp.") or std.mem.indexOf(u8, key, ".mtp.") != null) return null;
    const parts = [_][]const u8{ ".weight", ".scales", ".biases" };
    const projs = [_][]const u8{ ".gate_proj", ".up_proj", ".down_proj" };
    // DeepSeek V4's spelling of gate, up, down.
    const v4_projs = [_][]const u8{ ".w1", ".w3", ".w2" };
    var rest = key;
    const part = for (parts, 0..) |p, i| {
        if (std.mem.endsWith(u8, rest, p)) break i;
    } else return null;
    rest = rest[0 .. rest.len - parts[part].len];
    const proj, const v4 = for (projs, v4_projs, 0..) |p, q, i| {
        if (std.mem.endsWith(u8, rest, p)) break .{ i, false };
        if (std.mem.endsWith(u8, rest, q)) break .{ i, true };
    } else return null;
    rest = rest[0 .. rest.len - (if (v4) v4_projs else projs)[proj].len];
    const comp: u8 = @intCast(proj * 3 + part);
    if (v4) return if (std.mem.endsWith(u8, rest, ".experts")) .{ .prefix = rest[0 .. rest.len - ".experts".len], .expert = null, .comp = comp } else null;
    if (std.mem.endsWith(u8, rest, ".switch_mlp")) return .{ .prefix = rest[0 .. rest.len - ".switch_mlp".len], .expert = null, .comp = comp };
    const at = std.mem.lastIndexOf(u8, rest, ".experts.") orelse return null;
    const e = std.fmt.parseInt(u32, rest[at + ".experts.".len ..], 10) catch return null;
    return .{ .prefix = rest[0..at], .expert = e, .comp = comp };
}

/// The number after `layers.` in `prefix` (max when none).
fn layerNumber(prefix: []const u8) u64 {
    const at = std.mem.lastIndexOf(u8, prefix, "layers.") orelse return std.math.maxInt(u64);
    var end = at + "layers.".len;
    while (end < prefix.len and std.ascii.isDigit(prefix[end])) end += 1;
    return std.fmt.parseInt(u64, prefix[at + "layers.".len .. end], 10) catch std.math.maxInt(u64);
}

/// Routed pairs a call takes at most.
pub const max_call_ids: usize = 1 << 14;

/// One call's slots: per routed pair its bank (`expert.BankKind` order) and row, and each bank's arrays. Valid until
/// the next `fetch`.
pub const Fetched = struct { kind: []const u8, row: []const u32, banks: [3]?BankArrays };

/// The stream over a pack's routed experts: a layer call's distinct experts (at most `S.max_route_ids`) made
/// resident in slot rows, each routed pair's bank and row; in the decode phase from the start, as the experts.bin
/// source runs.
pub const Source = struct {
    a: std.mem.Allocator,
    bank: Bank,
    stream: *S.Stream,
    rows: []u32,
    kind: [max_call_ids]u8 = undefined,
    row: [max_call_ids]u32 = undefined,

    /// The pack in `dir` with `rows` slot rows per layer (null: every expert), slot arrays on `s`. Takes the
    /// process's one expert reader (refused by name while another holds it).
    pub fn open(a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream, rows: ?u32) !*Source {
        try expert.takeReader();
        errdefer expert.giveReader();
        var bank = try Bank.open(a, dir);
        errdefer bank.deinit();
        const per = rows orelse bank.n_experts;
        if (per == 0 or per > bank.n_experts) return error.StreamedRows;
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .bank = bank, .stream = undefined, .rows = try a.alloc(u32, bank.layers.len) };
        errdefer a.free(self.rows);
        @memset(self.rows, per);
        self.stream = try S.Stream.init(a, &self.bank, .{
            .rows = self.rows,
            .transient_rows = S.max_route_ids,
            .records_per_part = 2,
            .pool = .{ .workers = 4, .tickets = 512 },
            .staging_from_bank = true,
            .slot_memory = .{ .mlx = s },
        });
        errdefer self.stream.deinit();
        try self.stream.grow(self.rows);
        return self;
    }

    pub fn deinit(self: *Source) void {
        self.stream.deinit();
        self.bank.deinit();
        self.a.free(self.rows);
        self.a.destroy(self);
        expert.giveReader();
    }

    /// Routed layer `layer`'s pairs `ids` (the router's flat order) made resident, their reads waited on the host. The
    /// previous call's slots are released first (`ids` came from an eval that consumed them).
    pub fn fetch(self: *Source, layer: u32, ids: []const u16) !Fetched {
        if (ids.len > max_call_ids) return error.StreamedCallTooWide;
        try self.stream.flush();
        var distinct: [S.max_route_ids]u16 = undefined;
        var at: [1 << 16]u8 = undefined;
        var n: usize = 0;
        for (ids) |e| {
            if (e >= self.bank.n_experts) return error.StreamedExpertId;
            if (std.mem.indexOfScalar(u16, distinct[0..n], e) != null) continue;
            if (n == distinct.len) return error.StreamedCallTooWide;
            distinct[n] = e;
            n += 1;
        }
        const r = try self.stream.route(layer, distinct[0..n], &.{});
        var refs: [S.max_route_ids]expert.SlotRef = undefined;
        const rf = self.stream.refsOf(r, &refs);
        for (0..r.n_parts) |p| {
            try self.stream.waitGu(r, @intCast(p));
            try self.stream.waitDown(r, @intCast(p));
        }
        for (distinct[0..n], 0..) |e, i| at[e] = @intCast(i);
        for (ids, 0..) |e, i| {
            self.kind[i] = @intFromEnum(rf[at[e]].bank);
            self.row[i] = rf[at[e]].row;
        }
        self.stream.release(r);
        var banks: [3]?BankArrays = @splat(null);
        for ([_]expert.BankKind{ .base, .ext, .transient }, &banks) |k, *b| b.* = self.stream.bankArrays(layer, k);
        return .{ .kind = self.kind[0..ids.len], .row = self.row[0..ids.len], .banks = banks };
    }

    /// After the eval that consumed the last call's slots.
    pub fn settle(self: *Source) !void {
        try self.stream.flush();
    }

    pub fn stats(self: *Source) expert.Stats {
        return self.stream.stats();
    }
};

// ── tests ──

const testing = std.testing;

test "mlx pack: routed expert names, stacked and per expert; the MTP head's and other tensors are not" {
    const st = routedOf("language_model.model.layers.3.mlp.switch_mlp.down_proj.biases").?;
    try testing.expectEqualStrings("language_model.model.layers.3.mlp", st.prefix);
    try testing.expectEqual(@as(?u32, null), st.expert);
    try testing.expectEqual(@as(u8, 8), st.comp);
    const pe = routedOf("model.layers.12.mlp.experts.41.up_proj.scales").?;
    try testing.expectEqualStrings("model.layers.12.mlp", pe.prefix);
    try testing.expectEqual(@as(?u32, 41), pe.expert);
    try testing.expectEqual(@as(u8, 4), pe.comp);
    const ds = routedOf("language_model.layers.5.ffn.experts.w3.scales").?;
    try testing.expectEqualStrings("language_model.layers.5.ffn", ds.prefix);
    try testing.expectEqual(@as(?u32, null), ds.expert);
    try testing.expectEqual(@as(u8, 4), ds.comp);
    try testing.expectEqual(@as(u8, 6), routedOf("language_model.layers.5.ffn.experts.w2.weight").?.comp);
    for ([_][]const u8{ "language_model.layers.5.ffn.shared_experts.w1.weight", "language_model.mtp.0.ffn.experts.w1.weight", "mtp.layers.0.mlp.switch_mlp.gate_proj.weight", "model.layers.0.mlp.shared_expert.gate_proj.weight", "model.layers.0.mlp.gate.weight", "model.layers.0.mlp.switch_mlp.gate_proj.bias" }) |k|
        try testing.expect(routedOf(k) == null);
    try testing.expectEqual(@as(u64, 12), layerNumber("model.layers.12.mlp"));
}
