const std = @import("std");

pub const EmbeddedSpec = struct { rows: u64, dim: u32, shards: u32, layer_index: u32 = 1 };
pub const EmbeddedInfo = struct { payload_bytes: u64, rows: RowDtype };
const prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.";
const layer_prefix = "language_model.model.layers.";
const scale_marker = ".ple.ple_embedding.ngram_embedding.weight_scale";
const Part = enum(u2) { weight, scales, biases };

/// The two checkpoint spellings of the sharded table, told apart by which one the index lists. oMLX packs:
/// 4-bit rows, BF16 scales/biases, an optional global `weight_scale`. JANG packs (JANGH4): MLX-affine 8-bit
/// rows with F16 scales/biases, served as f16 rows, beside the hash constants the converter recorded.
pub const Layout = enum {
    omlx,
    jang,

    fn layerPrefix(self: Layout) []const u8 {
        return switch (self) {
            .omlx => layer_prefix,
            .jang => "language_model.layers.",
        };
    }

    fn shardMarker(self: Layout) []const u8 {
        return switch (self) {
            .omlx => ".ple.ple_embedding.ngram_embedding.shards.",
            .jang => ".ple.ngram_embedding.shards.",
        };
    }

    fn scalesDtype(self: Layout) []const u8 {
        return switch (self) {
            .omlx => "BF16",
            .jang => "F16",
        };
    }

    fn bits(self: Layout) u32 {
        return switch (self) {
            .omlx => 4,
            .jang => 8,
        };
    }

    /// The dtype a gathered row is served in: a JANG pack's are f16 like its scales (the dtype vMLX feeds
    /// the PLE block), every other table's bf16.
    pub fn rowDtype(self: Layout) RowDtype {
        return switch (self) {
            .omlx => .bf16,
            .jang => .f16,
        };
    }
};

pub const RowDtype = enum { bf16, f16 };

/// A JANG pack's I64 hash constants beside the table (`language_model.layers.N.ple.<name>`);
/// `NgramTable.verifyHashBuffers` compares them with the runtime hash.
pub const HashBuffer = enum { layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets };

const Key = struct { layer: u32, shard: u32, part: Part };

fn parseKey(name: []const u8, layout: Layout) ?Key {
    const lp = layout.layerPrefix();
    const sm = layout.shardMarker();
    if (!std.mem.startsWith(u8, name, lp)) return null;
    const tail = name[lp.len..];
    const marker = std.mem.indexOf(u8, tail, sm) orelse return null;
    if (marker == 0 or (marker > 1 and tail[0] == '0')) return null;
    const layer = std.fmt.parseInt(u32, tail[0..marker], 10) catch return null;
    const rest = tail[marker + sm.len ..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    if (dot == 0 or (dot > 1 and rest[0] == '0')) return null;
    const shard = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    const part: Part = if (std.mem.eql(u8, rest[dot + 1 ..], "weight")) .weight else if (std.mem.eql(u8, rest[dot + 1 ..], "scales")) .scales else if (std.mem.eql(u8, rest[dot + 1 ..], "biases")) .biases else return null;
    return .{ .layer = layer, .shard = shard, .part = part };
}

fn isTableKey(name: []const u8, layout: Layout) bool {
    return std.mem.startsWith(u8, name, layout.layerPrefix()) and std.mem.indexOf(u8, name, layout.shardMarker()) != null;
}

fn isScaleKey(name: []const u8) bool {
    return std.mem.startsWith(u8, name, layer_prefix) and std.mem.endsWith(u8, name, scale_marker);
}

/// The layout the index lists; null when it lists no table shard. A JANG table beside any oMLX table
/// tensor is ambiguous and refused.
fn detectLayout(weights: std.json.ObjectMap) !?Layout {
    var omlx = false;
    var jang = false;
    var scale = false;
    var it = weights.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        omlx = omlx or isTableKey(name, .omlx);
        jang = jang or isTableKey(name, .jang);
        scale = scale or isScaleKey(name);
    }
    if (jang and (omlx or scale)) return error.EmbeddedPleAmbiguous;
    return if (jang) .jang else if (omlx) .omlx else null;
}

pub fn embeddedTensorName(name: []const u8) bool {
    return parseKey(name, .omlx) != null or parseKey(name, .jang) != null or isScaleKey(name);
}

pub fn regionsOverlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) bool {
    return a_start < b_end and b_start < a_end;
}

pub const HeaderRegion = struct {
    rows: u64,
    cols: u64,
    start: u64,
    end: u64,

    pub fn overlaps(a: HeaderRegion, b: HeaderRegion) bool {
        return regionsOverlap(a.start, a.end, b.start, b.end);
    }
};

/// `vector` is one 1-D tensor of any length, read as a single row.
pub const RegionShape = enum { matrix, scalar, vector };

pub fn headerRegion(obj: std.json.ObjectMap, name: []const u8, dtype: []const u8, elem: u64, map_len: usize, data_off: usize, shape_kind: RegionShape) !HeaderRegion {
    const v = obj.get(name) orelse return error.TensorHeader;
    if (v != .object) return error.TensorHeader;
    const dt = v.object.get("dtype") orelse return error.TensorHeader;
    if (dt != .string or !std.mem.eql(u8, dt.string, dtype)) return error.TensorDtype;
    const shape = v.object.get("shape") orelse return error.TensorHeader;
    const offsets = v.object.get("data_offsets") orelse return error.TensorHeader;
    if (shape != .array or offsets != .array or offsets.array.items.len != 2) return error.TensorHeader;
    const dims = shape.array.items;
    if (shape_kind == .matrix and dims.len != 2) return error.TensorHeader;
    if (shape_kind == .scalar and dims.len != 0 and dims.len != 1) return error.TensorHeader;
    if (shape_kind == .vector and dims.len != 1) return error.TensorHeader;
    const o = offsets.array.items;
    for (o) |x| if (x != .integer) return error.TensorHeader;
    for (dims) |x| if (x != .integer) return error.TensorHeader;
    const rows_i: i64 = if (shape_kind != .matrix) 1 else dims[0].integer;
    const cols_i: i64 = if (shape_kind == .scalar and dims.len == 0) 1 else if (shape_kind != .matrix) dims[0].integer else dims[1].integer;
    if (rows_i <= 0 or cols_i <= 0 or o[0].integer < 0 or o[1].integer < o[0].integer) return error.TensorRegion;
    if (shape_kind == .scalar and cols_i != 1) return error.TensorRegion;
    const r: HeaderRegion = .{ .rows = @intCast(rows_i), .cols = @intCast(cols_i), .start = @intCast(o[0].integer), .end = @intCast(o[1].integer) };
    const need = std.math.mul(u64, std.math.mul(u64, r.rows, r.cols) catch return error.TensorRegion, elem) catch return error.TensorRegion;
    if (r.end - r.start != need) return error.TensorRegion;
    const abs_end = std.math.add(u64, data_off, r.end) catch return error.TensorTruncated;
    if (abs_end > map_len) return error.TensorTruncated;
    return r;
}

const Candidate = struct {
    names: [3]?[]const u8 = .{ null, null, null },
    files: [3]?[]const u8 = .{ null, null, null },
};

const Tensor = struct {
    file: usize,
    off: usize,
    len: usize,
    rows: u64,
    cols: u32,
};

const Shard = struct {
    first: u64,
    rows: u64,
    parts: [3]Tensor,
};

const File = struct {
    fd: std.c.fd_t,
    map: []align(std.heap.page_size_min) const u8,
    obj: std.json.ObjectMap,
    data_off: usize,
};

pub const RowParts = struct {
    weight: []const u8,
    scales: []const u8,
    biases: []const u8,
};

pub const Region = struct { fd: std.c.fd_t, off: u64, len: u64 };

pub const EmbeddedTable = struct {
    arena: std.heap.ArenaAllocator,
    files: []File,
    shards: []Shard,
    rows: u64,
    dim: u32,
    bits: u32,
    group_size: u32,
    wcols: u32,
    scols: u32,
    payload_bytes: u64,
    /// The pack's global `weight_scale`: oMLX stores the table unscaled and
    /// multiplies every gathered row by it (1.0 when absent).
    scale: f32 = 1.0,
    layout: Layout = .omlx,
    /// A JANG pack's `HashBuffer` tensors, I64 vectors in `HashBuffer` order (always set for `.jang`).
    hash_buffers: [3]?Tensor = .{ null, null, null },

    /// The raw little-endian I64 bytes of one hash constant tensor, null when the pack has none.
    pub fn hashBuffer(self: *const EmbeddedTable, which: HashBuffer) ?[]const u8 {
        const t = self.hash_buffers[@backingInt(which)] orelse return null;
        return self.files[t.file].map[t.off..][0..t.len];
    }

    pub fn close(self: *EmbeddedTable) void {
        for (self.files) |f| {
            std.posix.munmap(f.map);
            _ = std.c.close(f.fd);
        }
        self.arena.deinit();
    }

    fn locate(self: *const EmbeddedTable, row: u64) struct { shard: *const Shard, local: u64 } {
        std.debug.assert(row < self.rows);
        var lo: usize = 0;
        var hi: usize = self.shards.len;
        while (lo + 1 < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.shards[mid].first <= row) lo = mid else hi = mid;
        }
        return .{ .shard = &self.shards[lo], .local = row - self.shards[lo].first };
    }

    pub fn rowParts(self: *const EmbeddedTable, row: u64) RowParts {
        const where = self.locate(row);
        const p = where.shard.parts;
        const wlen: usize = self.wcols * 4;
        const slen: usize = self.scols * 2;
        return .{
            .weight = self.files[p[0].file].map[p[0].off + where.local * wlen ..][0..wlen],
            .scales = self.files[p[1].file].map[p[1].off + where.local * slen ..][0..slen],
            .biases = self.files[p[2].file].map[p[2].off + where.local * slen ..][0..slen],
        };
    }

    /// The `i`-th byte range a gather reads (each shard's three parts), null past the last.
    pub fn region(self: *const EmbeddedTable, i: usize) ?Region {
        if (i >= self.shards.len * 3) return null;
        const t = self.shards[i / 3].parts[i % 3];
        return .{ .fd = self.files[t.file].fd, .off = t.off, .len = t.len };
    }

    /// Bytes of the contiguous `weight | scales | biases` layout `repackInto` writes.
    pub fn repackedLen(self: *const EmbeddedTable) u64 {
        return self.rows * (@as(u64, self.wcols) * 4 + @as(u64, self.scols) * 4);
    }

    /// Copy every shard into `dst` as one `weight | scales | biases` table by global row, the
    /// layout of `ngram_table.bin`. Reads bypass the page cache: the copy is the resident one.
    pub fn repackInto(self: *const EmbeddedTable, dst: []u8) !void {
        if (dst.len < self.repackedLen()) return error.EmbeddedPleRepackShort;
        const wl: u64 = @as(u64, self.wcols) * 4;
        const sl: u64 = @as(u64, self.scols) * 2;
        const base = [3]u64{ 0, self.rows * wl, self.rows * (wl + sl) };
        const row_len = [3]u64{ wl, sl, sl };
        for (self.shards) |shard| for (shard.parts, 0..) |t, p| {
            const fd = self.files[t.file].fd;
            const out = dst[@intCast(base[p] + shard.first * row_len[p])..][0..t.len];
            // F_NOCACHE is Darwin-only; Linux reads through the page cache.
            const nocache = comptime @import("builtin").os.tag.isDarwin();
            if (nocache) _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
            defer if (nocache) {
                _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 0));
            };
            var done: usize = 0;
            while (done < out.len) {
                const want = @min(out.len - done, 1 << 30);
                const got = std.c.pread(fd, out[done..].ptr, want, @intCast(t.off + done));
                if (got <= 0) return error.EmbeddedPleRead;
                done += @intCast(got);
            }
        };
    }

    pub fn preadPart(self: *const EmbeddedTable, row: u64, part: usize, dst: []u8) bool {
        const where = self.locate(row);
        const t = where.shard.parts[part];
        if (dst.len != t.len / t.rows) return false;
        const off = t.off + where.local * dst.len;
        return std.c.pread(self.files[t.file].fd, dst.ptr, dst.len, @intCast(off)) == @as(isize, @intCast(dst.len));
    }
};

fn validFileName(name: []const u8) bool {
    return name.len > 0 and std.mem.endsWith(u8, name, ".safetensors") and std.mem.indexOfAny(u8, name, "/\\") == null and !std.mem.eql(u8, name, ".") and !std.mem.eql(u8, name, "..");
}

fn openFile(a: std.mem.Allocator, model_dir: []const u8, name: []const u8) !File {
    if (!validFileName(name)) return error.EmbeddedPleIndex;
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const full = std.fmt.bufPrint(&path, "{s}/{s}", .{ model_dir, name }) catch return error.NameTooLong;
    if (full.len >= path.len) return error.NameTooLong;
    path[full.len] = 0;
    const fd = std.c.open(path[0..full.len :0], .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.FileNotFound;
    errdefer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 8) return error.EmbeddedPleHeader;
    const size: usize = @intCast(end);
    const map = try std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, fd, 0);
    errdefer std.posix.munmap(map);
    const hlen = std.mem.readInt(u64, map[0..8], .little);
    if (hlen == 0 or hlen > 4 * 1024 * 1024 or hlen > size - 8) return error.EmbeddedPleHeader;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, map[8 .. 8 + @as(usize, @intCast(hlen))], .{}) catch return error.EmbeddedPleHeader;
    if (parsed != .object) return error.EmbeddedPleHeader;
    return .{ .fd = fd, .map = map, .obj = parsed.object, .data_off = 8 + @as(usize, @intCast(hlen)) };
}

fn regionOf(file: File, name: []const u8, dtype: []const u8, elem: u64, shape: RegionShape) !HeaderRegion {
    return headerRegion(file.obj, name, dtype, elem, file.map.len, file.data_off, shape) catch |err| return switch (err) {
        error.TensorHeader => error.EmbeddedPleHeader,
        error.TensorDtype => error.EmbeddedPleDtype,
        error.TensorRegion, error.TensorTruncated => error.EmbeddedPleRegion,
    };
}

fn tensor(file: File, file_id: usize, name: []const u8, dtype: []const u8, elem: u64, shape: RegionShape) !Tensor {
    const region = try regionOf(file, name, dtype, elem, shape);
    if (region.cols > std.math.maxInt(u32)) return error.EmbeddedPleRegion;
    return .{ .file = file_id, .off = file.data_off + @as(usize, @intCast(region.start)), .len = @intCast(region.end - region.start), .rows = region.rows, .cols = @intCast(region.cols) };
}

/// The pack files a table reads, each opened (and mapped) once.
const Files = struct {
    list: std.ArrayList(File) = .empty,
    ids: std.StringHashMapUnmanaged(usize) = .empty,

    fn id(self: *Files, a: std.mem.Allocator, model_dir: []const u8, name: []const u8) !usize {
        if (self.ids.get(name)) |existing| return existing;
        const f = try openFile(a, model_dir, name);
        self.list.append(a, f) catch |err| {
            std.posix.munmap(f.map);
            _ = std.c.close(f.fd);
            return err;
        };
        const next = self.list.items.len - 1;
        try self.ids.put(a, name, next);
        return next;
    }

    fn close(self: *Files) void {
        for (self.list.items) |f| {
            std.posix.munmap(f.map);
            _ = std.c.close(f.fd);
        }
    }
};

fn overlapsAny(shards: []const Shard, t: Tensor) bool {
    for (shards) |shard| for (shard.parts) |p| {
        if (p.file == t.file and regionsOverlap(t.off, t.off + t.len, p.off, p.off + p.len)) return true;
    };
    return false;
}

pub fn openEmbedded(model_dir: []const u8, expected: EmbeddedSpec) !?EmbeddedTable {
    if (expected.rows == 0 or expected.dim == 0 or expected.shards == 0 or expected.shards > 1024) return error.EmbeddedPleSpec;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, model_dir, .{});
    defer dir.close(io);
    const index = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 * 1024 * 1024)) catch return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, index, .{}) catch return error.EmbeddedPleIndex;
    if (parsed != .object) return error.EmbeddedPleIndex;
    const weights = parsed.object.get("weight_map") orelse return error.EmbeddedPleIndex;
    if (weights != .object) return error.EmbeddedPleIndex;
    const layout = (try detectLayout(weights.object)) orelse {
        arena.deinit();
        return null;
    };
    const candidates = try a.alloc(Candidate, expected.shards);
    for (candidates) |*c| c.* = .{};
    const scale_name = try std.fmt.allocPrint(a, "{s}{d}{s}", .{ layer_prefix, expected.layer_index, scale_marker });
    var scale_file: ?[]const u8 = null;
    var buffer_names: [3][]const u8 = undefined;
    for (&buffer_names, std.enums.values(HashBuffer)) |*n, which| {
        n.* = try std.fmt.allocPrint(a, "{s}{d}.ple.{s}", .{ Layout.jang.layerPrefix(), expected.layer_index, @tagName(which) });
    }
    var buffer_files: [3]?[]const u8 = .{ null, null, null };
    var found: usize = 0;
    var it = weights.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (isScaleKey(name)) {
            if (!std.mem.eql(u8, name, scale_name) or entry.value_ptr.* != .string) return error.EmbeddedPleIndex;
            scale_file = entry.value_ptr.string;
            continue;
        }
        if (layout == .jang) {
            const buffer: ?usize = for (buffer_names, 0..) |bn, j| {
                if (std.mem.eql(u8, name, bn)) break j;
            } else null;
            if (buffer) |j| {
                if (entry.value_ptr.* != .string) return error.EmbeddedPleIndex;
                buffer_files[j] = entry.value_ptr.string;
                continue;
            }
        }
        if (!isTableKey(name, layout)) continue;
        const key = parseKey(name, layout) orelse return error.EmbeddedPleIndex;
        if (key.layer != expected.layer_index or key.shard >= expected.shards or entry.value_ptr.* != .string) return error.EmbeddedPleIndex;
        const part = @backingInt(key.part);
        const c = &candidates[key.shard];
        if (c.names[part] != null) return error.EmbeddedPleDuplicate;
        c.names[part] = name;
        c.files[part] = entry.value_ptr.string;
        found += 1;
    }
    if (found != @as(usize, expected.shards) * 3) return error.EmbeddedPleMissing;
    if (layout == .jang) {
        for (buffer_files) |f| if (f == null) return error.EmbeddedPleHashMissing;
    }
    var files: Files = .{};
    errdefer files.close();
    const shards = try a.alloc(Shard, expected.shards);
    var first: u64 = 0;
    var payload: u64 = 0;
    var wcols: u32 = 0;
    var scols: u32 = 0;
    var group_size: u32 = 0;
    for (candidates, 0..) |c, i| {
        var parts: [3]Tensor = undefined;
        for (0..3) |p| {
            const name = c.names[p] orelse return error.EmbeddedPleMissing;
            const id = try files.id(a, model_dir, c.files[p] orelse return error.EmbeddedPleMissing);
            parts[p] = try tensor(files.list.items[id], id, name, if (p == 0) "U32" else layout.scalesDtype(), if (p == 0) 4 else 2, .matrix);
            payload = std.math.add(u64, payload, parts[p].len) catch return error.EmbeddedPleRegion;
        }
        if (parts[0].rows != parts[1].rows or parts[0].rows != parts[2].rows or parts[1].cols != parts[2].cols) return error.EmbeddedPleRegion;
        if (@as(u64, parts[0].cols) * 32 != @as(u64, expected.dim) * layout.bits() or parts[1].cols == 0 or expected.dim % parts[1].cols != 0) return error.EmbeddedPleGeometry;
        const gs = expected.dim / parts[1].cols;
        if (gs == 0 or gs > 1024 or (i != 0 and (parts[0].cols != wcols or parts[1].cols != scols or gs != group_size))) return error.EmbeddedPleGeometry;
        // Row r is at shard r / per, per = shard 0's rows: every shard but the last holds exactly per.
        if (layout == .jang and i != 0) {
            const per = shards[0].rows;
            if (if (i + 1 == candidates.len) parts[0].rows > per else parts[0].rows != per) return error.EmbeddedPleGeometry;
        }
        for (parts, 0..) |p, x| for (parts[0..x]) |q| {
            if (p.file == q.file and regionsOverlap(p.off, p.off + p.len, q.off, q.off + q.len)) return error.EmbeddedPleOverlap;
        };
        shards[i] = .{ .first = first, .rows = parts[0].rows, .parts = parts };
        first = std.math.add(u64, first, parts[0].rows) catch return error.EmbeddedPleRegion;
        wcols = parts[0].cols;
        scols = parts[1].cols;
        group_size = gs;
    }
    if (first != expected.rows) return error.EmbeddedPleGeometry;
    for (shards, 0..) |shard, i| for (shard.parts) |p| {
        if (overlapsAny(shards[0..i], p)) return error.EmbeddedPleOverlap;
    };
    var scale: f32 = 1.0;
    if (scale_file) |file_name| {
        const id = try files.id(a, model_dir, file_name);
        const file = files.list.items[id];
        const t = try tensor(file, id, scale_name, "BF16", 2, .scalar);
        if (overlapsAny(shards, t)) return error.EmbeddedPleOverlap;
        const bits: u32 = std.mem.readInt(u16, file.map[t.off..][0..2], .little);
        scale = @bitCast(bits << 16);
        if (!std.math.isFinite(scale) or scale <= 0) return error.EmbeddedPleWeightScale;
        payload = std.math.add(u64, payload, t.len) catch return error.EmbeddedPleRegion;
    }
    var hash_buffers: [3]?Tensor = .{ null, null, null };
    if (layout == .jang) for (buffer_names, buffer_files, &hash_buffers) |name, file_name, *slot| {
        const id = try files.id(a, model_dir, file_name.?);
        const t = try tensor(files.list.items[id], id, name, "I64", 8, .vector);
        if (overlapsAny(shards, t)) return error.EmbeddedPleOverlap;
        slot.* = t;
    };
    return .{ .arena = arena, .files = files.list.items, .shards = shards, .rows = first, .dim = expected.dim, .bits = layout.bits(), .group_size = group_size, .wcols = wcols, .scols = scols, .payload_bytes = payload, .scale = scale, .layout = layout, .hash_buffers = hash_buffers };
}

pub fn inspectEmbedded(model_dir: []const u8, expected: EmbeddedSpec) !?EmbeddedInfo {
    var table = (try openEmbedded(model_dir, expected)) orelse return null;
    defer table.close();
    return .{ .payload_bytes = table.payload_bytes, .rows = table.layout.rowDtype() };
}

pub const FixtureVariant = enum { valid, missing, duplicate, extra, dtype, bounds, overlap, scale_unit, scale_scalar, scale_nonunit, scale_dtype, scale_shape, scale_overlap };

pub fn writeFixture(td: *std.testing.TmpDir, variant: FixtureVariant) !void {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const index = try std.fmt.allocPrint(a, "{{\"weight_map\":{{\"{s}2.biases\":\"a.safetensors\",\"{s}0.weight\":\"a.safetensors\",\"{s}1.weight\":\"b.safetensors\",\"{s}2.weight\":\"a.safetensors\",\"{s}0.scales\":\"a.safetensors\",\"{s}1.scales\":\"b.safetensors\",\"{s}0.biases\":\"a.safetensors\",\"{s}2.scales\":\"a.safetensors\"{s}}}}}", .{ prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, if (variant == .missing) "" else if (variant == .duplicate) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight\":\"a.safetensors\",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\"" else if (variant == .extra) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.3.weight\":\"a.safetensors\"" else if (@backingInt(variant) >= @backingInt(FixtureVariant.scale_unit)) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":\"a.safetensors\"" else ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.1.biases\":\"b.safetensors\"" });
    defer a.free(index);
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    const header_a = try std.fmt.allocPrint(a, "{{\"{s}2.biases\":{{\"dtype\":\"BF16\",\"shape\":[3,1],\"data_offsets\":[0,6]}}," ++
        "\"{s}0.weight\":{{\"dtype\":\"{s}\",\"shape\":[2,4],\"data_offsets\":[6,38]}}," ++
        "\"{s}2.weight\":{{\"dtype\":\"U32\",\"shape\":[3,4],\"data_offsets\":[{d},{d}]}}," ++
        "\"{s}0.scales\":{{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[86,90]}}," ++
        "\"{s}0.biases\":{{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[{d},{d}]}}," ++
        "\"{s}2.scales\":{{\"dtype\":\"BF16\",\"shape\":[3,1],\"data_offsets\":[94,100]}}{s}}}", .{ prefix, prefix, if (variant == .dtype) "F32" else "U32", prefix, if (variant == .bounds) @as(u32, 999) else 38, if (variant == .bounds) @as(u32, 1047) else 86, prefix, prefix, if (variant == .overlap) @as(u32, 88) else 90, if (variant == .overlap) @as(u32, 92) else 94, prefix, if (@backingInt(variant) >= @backingInt(FixtureVariant.scale_unit)) if (variant == .scale_dtype) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[100,102]}" else if (variant == .scale_shape) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[100,102]}" else if (variant == .scale_scalar) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[100,102]}" else if (variant == .scale_overlap) ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[98,100]}" else ",\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[100,102]}" else "" });
    defer a.free(header_a);
    const header_b = try std.fmt.allocPrint(a, "{{\"{s}1.scales\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[0,2]}}," ++
        "\"{s}1.weight\":{{\"dtype\":\"U32\",\"shape\":[1,4],\"data_offsets\":[2,18]}}," ++
        "\"{s}1.biases\":{{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[18,20]}}}}", .{ prefix, prefix, prefix });
    defer a.free(header_b);
    var data_a: [102]u8 = @splat(0);
    @memset(data_a[6..22], 0x00);
    @memset(data_a[22..38], 0x11);
    @memset(data_a[38..54], 0x33);
    @memset(data_a[54..70], 0x44);
    @memset(data_a[70..86], 0x55);
    for ([_]usize{ 86, 88, 94, 96, 98 }) |off| std.mem.writeInt(u16, data_a[off..][0..2], 0x3f80, .little);
    std.mem.writeInt(u16, data_a[100..102], if (variant == .scale_nonunit) 0x4000 else 0x3f80, .little);
    var data_b: [20]u8 = @splat(0);
    std.mem.writeInt(u16, data_b[0..2], 0x3f80, .little);
    @memset(data_b[2..18], 0x22);
    try writeSafetensors(td, "a.safetensors", header_a, data_a[0..if (@backingInt(variant) >= @backingInt(FixtureVariant.scale_unit)) 102 else 100]);
    try writeSafetensors(td, "b.safetensors", header_b, &data_b);
}

fn writeSafetensors(td: *std.testing.TmpDir, name: []const u8, header: []const u8, data: []const u8) !void {
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (header.len > 2048) return error.TestHeaderTooLarge;
    const buf = try a.alloc(u8, 8 + 2048 + data.len);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], 2048, .little);
    @memset(buf[8..2056], ' ');
    @memcpy(buf[8 .. 8 + header.len], header);
    @memcpy(buf[2056..], data);
    try td.dir.writeFile(io, .{ .sub_path = name, .data = buf });
}

pub const JangVariant = enum { valid, scales_bf16, bits4, unequal, last_longer, missing_buffer, buffer_dtype, ambiguous };

/// The constants a JANG fixture records as its `HashBuffer`s.
pub const JangHash = struct { multipliers: []const i64, vocab: []const i64, offsets: []const i64 };

pub const JANG_FIXTURE_SPEC: EmbeddedSpec = .{ .rows = 5, .dim = 32, .shards = 3 };

/// Weight byte `i` of global row `g` in a JANG fixture: identifies the row, and the rows span all 256 codes.
pub fn jangFixtureCode(g: u64, i: usize) u8 {
    return @truncate(g * 37 + i * 11);
}

/// Scales and biases cover subnormal and normal halves of either sign, products past f16's range included.
fn randHalf(r: std.Random) u16 {
    const sign: u16 = @as(u16, r.int(u1)) << 15;
    const exp: u16 = if (r.uintLessThan(u8, 8) == 0) 0 else r.intRangeAtMost(u16, 1, 24);
    return sign | (exp << 10) | r.int(u10);
}

/// Tensors appended into two safetensors files and their index, in order.
const FixtureFiles = struct {
    const names = [2][]const u8{ "a.safetensors", "b.safetensors" };
    headers: [2]std.ArrayList(u8) = .{ .empty, .empty },
    data: [2]std.ArrayList(u8) = .{ .empty, .empty },
    index: std.ArrayList(u8) = .empty,

    fn deinit(self: *FixtureFiles, a: std.mem.Allocator) void {
        for (&self.headers, &self.data) |*h, *d| {
            h.deinit(a);
            d.deinit(a);
        }
        self.index.deinit(a);
    }

    fn add(self: *FixtureFiles, a: std.mem.Allocator, file: usize, name: []const u8, dtype: []const u8, shape: []const u64, bytes: []const u8) !void {
        const h = &self.headers[file];
        try h.appendSlice(a, if (h.items.len == 0) "{" else ",");
        try h.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ name, dtype });
        for (shape, 0..) |d, i| try h.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
        const start = self.data[file].items.len;
        try h.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ start, start + bytes.len });
        try self.data[file].appendSlice(a, bytes);
        try self.list(a, file, name);
    }

    fn list(self: *FixtureFiles, a: std.mem.Allocator, file: usize, name: []const u8) !void {
        try self.index.appendSlice(a, if (self.index.items.len == 0) "{\"weight_map\":{" else ",");
        try self.index.print(a, "\"{s}\":\"{s}\"", .{ name, names[file] });
    }

    fn write(self: *FixtureFiles, a: std.mem.Allocator, td: *std.testing.TmpDir) !void {
        for (&self.headers, self.data, names) |*h, d, name| {
            try h.append(a, '}');
            try writeSafetensors(td, name, h.items, d.items);
        }
        try self.index.appendSlice(a, "}}");
        const io = std.Io.Threaded.global_single_threaded.io();
        try td.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = self.index.items });
    }
};

/// A JANG-layout pack in JANGH4's spelling (`JANG_FIXTURE_SPEC`: dim 32, 8-bit, one F16 scale and bias per
/// row): shards 0 and 2 and the hash buffers in a.safetensors, shard 1 in b.safetensors, 2/2/1 rows. Weight
/// bytes are `jangFixtureCode`; scales and biases come from `seed`.
pub fn writeJangFixture(td: *std.testing.TmpDir, variant: JangVariant, hash: JangHash, seed: u64) !void {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    var ff: FixtureFiles = .{};
    defer ff.deinit(a);
    const rows: [3]u64 = switch (variant) {
        .unequal => .{ 2, 1, 2 },
        .last_longer => .{ 1, 1, 3 },
        else => .{ 2, 2, 1 },
    };
    const wbytes: usize = if (variant == .bits4) 16 else 32;
    var name_buf: [128]u8 = undefined;
    var g: u64 = 0;
    for (rows, 0..) |n, s| {
        const file: usize = if (s == 1) 1 else 0;
        var w: [3 * 32]u8 = undefined;
        var sc: [3 * 2]u8 = undefined;
        var bi: [3 * 2]u8 = undefined;
        for (0..n) |l| {
            for (0..wbytes) |i| w[l * wbytes + i] = jangFixtureCode(g + l, i);
            std.mem.writeInt(u16, sc[l * 2 ..][0..2], randHalf(r), .little);
            std.mem.writeInt(u16, bi[l * 2 ..][0..2], randHalf(r), .little);
        }
        g += n;
        const sdt = if (variant == .scales_bf16) "BF16" else "F16";
        try ff.add(a, file, try std.fmt.bufPrint(&name_buf, "language_model.layers.1.ple.ngram_embedding.shards.{d}.weight", .{s}), "U32", &.{ n, wbytes / 4 }, w[0 .. n * wbytes]);
        try ff.add(a, file, try std.fmt.bufPrint(&name_buf, "language_model.layers.1.ple.ngram_embedding.shards.{d}.scales", .{s}), sdt, &.{ n, 1 }, sc[0 .. n * 2]);
        try ff.add(a, file, try std.fmt.bufPrint(&name_buf, "language_model.layers.1.ple.ngram_embedding.shards.{d}.biases", .{s}), sdt, &.{ n, 1 }, bi[0 .. n * 2]);
    }
    for (std.enums.values(HashBuffer), [_][]const i64{ hash.multipliers, hash.vocab, hash.offsets }) |which, values| {
        if (variant == .missing_buffer and which == .ngram_heads_offsets) continue;
        var bytes: [8 * 32]u8 = undefined;
        const elem: usize = if (variant == .buffer_dtype) 4 else 8;
        for (values, 0..) |v, i| {
            if (elem == 4) std.mem.writeInt(i32, bytes[i * 4 ..][0..4], @intCast(v), .little) else std.mem.writeInt(i64, bytes[i * 8 ..][0..8], v, .little);
        }
        try ff.add(a, 0, try std.fmt.bufPrint(&name_buf, "language_model.layers.1.ple.{s}", .{@tagName(which)}), if (elem == 4) "I32" else "I64", &.{values.len}, bytes[0 .. values.len * elem]);
    }
    if (variant == .ambiguous) try ff.list(a, 0, prefix ++ "0.weight");
    try ff.write(a, td);
}

test "embedded PLE reads unequal numbered shards across mixed files" {
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeFixture(&td, .valid);
    const io = std.Io.Threaded.global_single_threaded.io();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try td.dir.realPath(io, &path_buf);
    const spec: EmbeddedSpec = .{ .rows = 6, .dim = 32, .shards = 3 };
    const info = (try inspectEmbedded(path_buf[0..path_len], spec)).?;
    try std.testing.expectEqual(@as(u64, 120), info.payload_bytes);
    var t = (try openEmbedded(path_buf[0..path_len], spec)).?;
    defer t.close();
    for ([_]u64{ 5, 1, 2, 0, 3, 5, 4 }) |r| {
        const parts = t.rowParts(r);
        try std.testing.expectEqual(@as(u8, @intCast(r * 0x11)), parts.weight[0]);
        try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, parts.scales[0..2], .little));
        var w: [16]u8 = undefined;
        var s: [2]u8 = undefined;
        var b: [2]u8 = undefined;
        try std.testing.expect(t.preadPart(r, 0, &w));
        try std.testing.expect(t.preadPart(r, 1, &s));
        try std.testing.expect(t.preadPart(r, 2, &b));
        try std.testing.expectEqualSlices(u8, parts.weight, &w);
        try std.testing.expectEqualSlices(u8, parts.scales, &s);
        try std.testing.expectEqualSlices(u8, parts.biases, &b);
    }
    try std.testing.expect(!embeddedTensorName(prefix ++ "00.weight"));
    try std.testing.expect(!embeddedTensorName("language_model.model.layers.1.ple.ple_embedding.layer_multipliers"));
    try std.testing.expect(embeddedTensorName("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"));
}

test "embedded PLE repacks every shard into one weight | scales | biases table by global row" {
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeFixture(&td, .valid);
    const io = std.Io.Threaded.global_single_threaded.io();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try td.dir.realPath(io, &path_buf);
    var t = (try openEmbedded(path_buf[0..path_len], .{ .rows = 6, .dim = 32, .shards = 3 })).?;
    defer t.close();
    try std.testing.expectEqual(@as(u64, 6 * (16 + 4)), t.repackedLen());
    var buf: [120]u8 = undefined;
    try std.testing.expectError(error.EmbeddedPleRepackShort, t.repackInto(buf[0..119]));
    try t.repackInto(&buf);
    for (0..6) |r| {
        const parts = t.rowParts(r);
        try std.testing.expectEqualSlices(u8, parts.weight, buf[r * 16 ..][0..16]);
        try std.testing.expectEqualSlices(u8, parts.scales, buf[96 + r * 2 ..][0..2]);
        try std.testing.expectEqualSlices(u8, parts.biases, buf[108 + r * 2 ..][0..2]);
    }
}

test "embedded PLE rejects partial and corrupt shard layouts" {
    const spec: EmbeddedSpec = .{ .rows = 6, .dim = 32, .shards = 3 };
    inline for (.{ .missing, .duplicate, .extra, .dtype, .bounds, .overlap }) |variant| {
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        try writeFixture(&td, variant);
        const io = std.Io.Threaded.global_single_threaded.io();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try td.dir.realPath(io, &path_buf);
        try std.testing.expectError(switch (variant) {
            .missing => error.EmbeddedPleMissing,
            .duplicate, .extra => error.EmbeddedPleIndex,
            .dtype => error.EmbeddedPleDtype,
            .bounds => error.EmbeddedPleRegion,
            .overlap => error.EmbeddedPleOverlap,
            else => unreachable,
        }, openEmbedded(path_buf[0..path_len], spec));
    }
}

test "embedded PLE reads the optional global weight scale and validates its tensor" {
    const spec: EmbeddedSpec = .{ .rows = 6, .dim = 32, .shards = 3 };
    inline for (.{ .valid, .scale_unit, .scale_scalar, .scale_nonunit, .scale_dtype, .scale_shape, .scale_overlap }) |variant| {
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        try writeFixture(&td, variant);
        const io = std.Io.Threaded.global_single_threaded.io();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try td.dir.realPath(io, &path_buf);
        const path = path_buf[0..path_len];
        if (variant == .valid or variant == .scale_unit or variant == .scale_scalar or variant == .scale_nonunit) {
            const info = (try inspectEmbedded(path, spec)).?;
            try std.testing.expectEqual(@as(u64, if (variant == .valid) 120 else 122), info.payload_bytes);
            var table = (try openEmbedded(path, spec)).?;
            defer table.close();
            try std.testing.expectEqual(@as(f32, if (variant == .scale_nonunit) 2.0 else 1.0), table.scale);
        } else {
            try std.testing.expectError(switch (variant) {
                .scale_dtype => error.EmbeddedPleDtype,
                .scale_shape => error.EmbeddedPleRegion,
                .scale_overlap => error.EmbeddedPleOverlap,
                else => unreachable,
            }, openEmbedded(path, spec));
        }
    }
}

test "global weight scale without shards leaves external table layout available" {
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeFixture(&td, .scale_unit);
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale\":\"a.safetensors\"}}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try td.dir.realPath(io, &path_buf);
    try std.testing.expect((try inspectEmbedded(path_buf[0..path_len], .{ .rows = 6, .dim = 32, .shards = 3 })) == null);
}

test "embedded PLE indexed checkpoint metadata validates without tensor reads" {
    const raw = std.c.getenv("QWEN4_EMBEDDED_TEST_MODEL") orelse return error.SkipZigTest;
    const path = std.mem.sliceTo(raw, 0);
    const info = (try inspectEmbedded(path, .{ .rows = 320_001_536, .dim = 160, .shards = 128 })).?;
    try std.testing.expectEqual(@as(u64, 32_000_153_602), info.payload_bytes);
}

const test_hash: JangHash = .{ .multipliers = &.{ 7, -9, 11 }, .vocab = &.{ 2, 3 }, .offsets = &.{ 0, 2 } };

fn jangFixturePath(td: *std.testing.TmpDir, variant: JangVariant, buf: *[std.fs.max_path_bytes]u8) ![]const u8 {
    try writeJangFixture(td, variant, test_hash, 5);
    const io = std.Io.Threaded.global_single_threaded.io();
    return buf[0..try td.dir.realPath(io, buf)];
}

test "jangh4 embedded PLE reads 8-bit F16 shards at vMLX's row r / per" {
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try jangFixturePath(&td, .valid, &path_buf);
    const info = (try inspectEmbedded(path, JANG_FIXTURE_SPEC)).?;
    try std.testing.expectEqual(@as(u64, 5 * (32 + 2 + 2)), info.payload_bytes);
    try std.testing.expectEqual(RowDtype.f16, info.rows);
    var t = (try openEmbedded(path, JANG_FIXTURE_SPEC)).?;
    defer t.close();
    try std.testing.expectEqual(Layout.jang, t.layout);
    try std.testing.expectEqual(@as(u32, 8), t.bits);
    try std.testing.expectEqual(@as(u32, 32), t.group_size);
    try std.testing.expectEqual(@as(f32, 1.0), t.scale);
    for ([_]u64{ 4, 0, 1, 2, 3, 4, 1 }) |r| {
        const parts = t.rowParts(r);
        for (parts.weight, 0..) |q, i| try std.testing.expectEqual(jangFixtureCode(r, i), q);
        var w: [32]u8 = undefined;
        var s: [2]u8 = undefined;
        var b: [2]u8 = undefined;
        try std.testing.expect(t.preadPart(r, 0, &w));
        try std.testing.expect(t.preadPart(r, 1, &s));
        try std.testing.expect(t.preadPart(r, 2, &b));
        try std.testing.expectEqualSlices(u8, parts.weight, &w);
        try std.testing.expectEqualSlices(u8, parts.scales, &s);
        try std.testing.expectEqualSlices(u8, parts.biases, &b);
    }
    for (std.enums.values(HashBuffer), [_][]const i64{ test_hash.multipliers, test_hash.vocab, test_hash.offsets }) |which, want| {
        const raw = t.hashBuffer(which).?;
        try std.testing.expectEqual(want.len * 8, raw.len);
        for (want, 0..) |v, i| try std.testing.expectEqual(v, std.mem.readInt(i64, raw[i * 8 ..][0..8], .little));
    }
    try std.testing.expect(embeddedTensorName("language_model.layers.1.ple.ngram_embedding.shards.127.scales"));
    try std.testing.expect(!embeddedTensorName("language_model.layers.1.ple.ngram_embedding.shards.07.weight"));
    try std.testing.expect(!embeddedTensorName("language_model.layers.1.ple.layer_multipliers"));
    try std.testing.expect(!embeddedTensorName("language_model.layers.1.ple.key_proj.weight"));
}

test "jangh4 embedded PLE refuses a pack it cannot serve exactly" {
    inline for (.{ .scales_bf16, .bits4, .unequal, .last_longer, .missing_buffer, .buffer_dtype, .ambiguous }) |variant| {
        var td = std.testing.tmpDir(.{});
        defer td.cleanup();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try jangFixturePath(&td, variant, &path_buf);
        try std.testing.expectError(switch (variant) {
            .scales_bf16, .buffer_dtype => error.EmbeddedPleDtype,
            .bits4, .unequal, .last_longer => error.EmbeddedPleGeometry,
            .missing_buffer => error.EmbeddedPleHashMissing,
            .ambiguous => error.EmbeddedPleAmbiguous,
            else => unreachable,
        }, openEmbedded(path, JANG_FIXTURE_SPEC));
    }
}
