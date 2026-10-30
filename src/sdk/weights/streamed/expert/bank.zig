//! Packed expert bank for the expert streamer: EXL3 v2 records in one
//! `experts.bin`, described by two manifests (v2 geometry + the v1 runtime
//! manifest). Every check runs once in `Bank.open`, which also opens the
//! sidecar the read pool reads through; the Python streaming stack's
//! first-boot checks (bankv2, exl3_lane) are their oracle.

const std = @import("std");
const env = @import("mlx_host").env;
const kset = @import("../sdk_ext/kernel_set.zig");
const io_util = @import("sdk").io_util;
const expert_io = @import("../sdk_ext.zig").expert.io;
const bank_file = @import("../sdk_ext.zig").expert.bank_file;
const mlx = @import("sdk").mlx;

/// Record segments in on-disk order: the gate/up span is the first six, the
/// down span the last three.
pub const Component = enum(u8) {
    gate_code,
    gate_rout,
    gate_rin,
    up_code,
    up_rout,
    up_rin,
    down_code,
    down_rout,
    down_rin,

    pub fn name(c: Component) []const u8 {
        return names[@intFromEnum(c)];
    }

    const names = [_][]const u8{
        "gate_proj.code", "gate_proj.rout", "gate_proj.rin",
        "up_proj.code",   "up_proj.rout",   "up_proj.rin",
        "down_proj.code", "down_proj.rout", "down_proj.rin",
    };
};
pub const n_components = 9;
pub const gu_components = 6;
/// A bank of K = 3 and K = 2 layers keeps a transient scratch per geometry (`sdk_ext.expert.stream`): a K = 2 layer's
/// rows land in K = 2-shaped slots in every bank (base, grown, transient), so the kernels read one tile width per
/// launch and a code array's tile width names its layer's K.
pub const per_geometry_transients = true;

pub const Dtype = enum { I16, F16 };

pub const Segment = struct {
    /// Relative to the record start.
    offset: u64,
    length: u64,
    dtype: Dtype,
    shape: [3]u64,
    rank: u8,
};

pub const Layer = struct {
    k: u32,
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};

/// What the installed kernels decode; a bank outside it is refused.
pub const Implemented = struct {
    codebooks: []const []const u8,
    k: []const u32,
    hidden: u64,
    inter: u64,
    n_experts: u64,
    n_layers: u64,
};

/// DeepSeek-V4.1 on an EXL3 bank: mul1, K = 3 (the 3.0 bpw bank, every layer) or K = 2 (the 2.9 bpw banks' layers 18-22).
pub const dsv41: Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{ 2, 3 }, .hidden = 5120, .inter = 2304, .n_experts = 384, .n_layers = 40 };

/// Why a bank was refused, for the one log line the caller writes.
pub const Diag = @import("sdk").Diag;

pub const Refusal = error{
    BankDirNotAbsolute,
    ManifestMissing,
    ManifestOpen,
    ManifestSyntax,
    ManifestFormat,
    CodebookNotImplemented,
    CodebookMultiplier,
    DimsInvalid,
    DimsNotImplemented,
    KNotImplemented,
    LayerGeometry,
    SidecarGeometry,
    SidecarMissing,
    SidecarOpen,
    RecordCount,
    RecordGeometry,
    RecordDuplicate,
    RecordSha256,
    ParityNotPassed,
    RuntimeModeNotExl3,
    RuntimeRecordMismatch,
};

pub const format_v2 = "mtplx-expert-manifest-v2";
pub const format_v1 = "mtplx-expert-manifest-v1";
/// The v1 runtime manifest's mode for an EXL3 bank (never a tcq3 / mcg bank).
pub const runtime_mode = "exl3-mul1";
/// Record alignment in experts.bin; the manifest must declare the same.
pub const record_alignment: u64 = 4096;
/// The mul1 codebook multiplier the EXL3 kernels are built for.
pub const mul1_multiplier: u64 = 0x83DCD12D;
/// Bounds the parse of a hostile manifest (the real bank has 40 x 384).
const max_layers = 4096;
const max_experts = 1 << 16;
const max_records = 1 << 20;

/// The EXL3 v2 record geometry at K (bankv2.layer_segments): per projection
/// `code` I16 [in/16, out/16, 16K], `rout` F16 [out], `rin` F16 [in]; gate/up
/// map hidden -> inter, down inter -> hidden; records pad to `record_alignment`.
/// A manifest whose layer table differs from this is refused.
pub fn layerSegments(k: u32, hidden: u64, inter: u64) ?Layer {
    if (k == 0 or k > 8 or hidden == 0 or inter == 0) return null;
    if (hidden % 16 != 0 or inter % 16 != 0 or hidden > 1 << 24 or inter > 1 << 24) return null;
    var l: Layer = .{ .k = k, .record_bytes = 0, .logical_bytes = 0, .base_offset = 0, .segments = undefined };
    var off: u64 = 0;
    for (0..3) |p| {
        const in: u64 = if (p == 2) inter else hidden;
        const out: u64 = if (p == 2) hidden else inter;
        const parts = [3]Segment{
            .{ .offset = 0, .length = (in / 16) * (out / 16) * 16 * k * 2, .dtype = .I16, .shape = .{ in / 16, out / 16, 16 * k }, .rank = 3 },
            .{ .offset = 0, .length = out * 2, .dtype = .F16, .shape = .{ out, 0, 0 }, .rank = 1 },
            .{ .offset = 0, .length = in * 2, .dtype = .F16, .shape = .{ in, 0, 0 }, .rank = 1 },
        };
        for (parts, 0..) |seg, j| {
            l.segments[p * 3 + j] = seg;
            l.segments[p * 3 + j].offset = off;
            off += seg.length;
        }
    }
    l.logical_bytes = off;
    l.record_bytes = std.mem.alignForward(u64, off, record_alignment);
    return l;
}

/// bankv2.layer_table: `out[L]` = the geometry at `ks[L]`, laid out layer-major,
/// expert-minor. Returns the bank's total bytes, or null (bad dims / overflow).
pub fn layerTable(ks: []const u32, hidden: u64, inter: u64, n_experts: u64, out: []Layer) ?u64 {
    var base: u64 = 0;
    for (ks, out) |k, *l| {
        l.* = layerSegments(k, hidden, inter) orelse return null;
        l.base_offset = base;
        const layer_bytes = std.math.mul(u64, n_experts, l.record_bytes) catch return null;
        base = std.math.add(u64, base, layer_bytes) catch return null;
    }
    return base;
}

/// A record's two read ranges; their parts are the layer's segment lengths.
pub const Spans = struct {
    /// Segments 0..5 (gate/up), contiguous from here.
    gu_offset: u64,
    /// Segments 6..8 (down), contiguous from here.
    down_offset: u64,
};

/// EXL3's record topology on the reader: nine components, six in the gate/up range.
pub const Records = expert_io.Records(n_components, gu_components);

pub const RecordRef = struct { layer: u32, expert: u32 };

/// `records` of `bank` (one segment geometry) as one job into `rows`.
pub fn submitRecords(pool: *expert_io.Pool, bank: *const Bank, records: []const RecordRef, rows: []const [n_components]u64) !u32 {
    if (records.len != rows.len or records.len == 0 or records.len > expert_io.max_items) return error.InvalidJob;
    var lens: [n_components]u64 = undefined;
    for (&lens, bank.layers[records[0].layer].segments) |*l, sg| l.* = sg.length;
    var gu: [expert_io.max_items]u64 = undefined;
    var down: [expert_io.max_items]u64 = undefined;
    for (records, 0..) |r, i| {
        for (lens, bank.layers[r.layer].segments) |l, sg| if (l != sg.length) return error.MixedGeometry;
        const sp = bank.spans(r.layer, r.expert);
        gu[i] = sp.gu_offset;
        down[i] = sp.down_offset;
    }
    return Records.submit(pool, bank.sidecar, gu[0..records.len], down[0..records.len], rows, &lens);
}

// The stream's bank contract (sdk_ext.expert.assertBank; sdk_ext.expert.stream): with the topology, `Component`, `Layer`,
// `Records` and `Bank` above and below, a segment's MLX dtype and the slot arrays by projection.

/// DeepSeek-V4.1 routes six experts per token.
pub const routed_top_k = 6;

pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .I16 => .int16,
        .F16 => .float16,
    };
}

/// One projection's slot arrays: code int16 [rows, in/16, out/16, 16K], rout f16 [rows, out], rin f16 [rows, in].
pub const ProjArrays = struct { code: mlx.mlx_array, rout: mlx.mlx_array, rin: mlx.mlx_array };
/// A bank's nine slot arrays by projection.
pub const BankArrays = struct { gate: ProjArrays, up: ProjArrays, down: ProjArrays };

pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{ .gate = .{ .code = x[0], .rout = x[1], .rin = x[2] }, .up = .{ .code = x[3], .rout = x[4], .rin = x[5] }, .down = .{ .code = x[6], .rout = x[7], .rin = x[8] } };
}

pub const Bank = struct {
    allocator: std.mem.Allocator,
    hidden: u64 = 0,
    inter: u64 = 0,
    n_experts: u32 = 0,
    layers: []Layer = &.{},
    /// [layer * n_experts + expert]
    digests: []Digests = &.{},
    sidecar_path: [:0]const u8 = "",
    /// experts.bin, opened once for the reader (`openUncached`: read-only, O_NOFOLLOW, F_NOCACHE, no read-ahead);
    /// `sidecar.size` is the file's.
    sidecar: expert_io.UncachedFd = .{ .fd = -1, .size = 0 },
    /// Bytes the records occupy (v2 `sidecar.size`, <= the file's).
    sidecar_size: u64 = 0,

    /// sha256 of the padded record (v2) and of its logical bytes (v1).
    pub const Digests = struct { padded: [32]u8, logical: [32]u8 };

    /// Parses and checks both manifests and opens experts.bin, once; any
    /// inconsistency refuses the whole bank (`diag` says which).
    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, implemented: Implemented, diag: ?*Diag) !Bank {
        // The absolute-path opens below assert this (UB in ReleaseFast).
        if (!std.fs.path.isAbsolute(dir)) return refuse(diag, error.BankDirNotAbsolute, "bank dir \"{s}\" is not an absolute path", .{dir});
        var bank: Bank = .{ .allocator = allocator };
        errdefer bank.deinit();
        var v2 = try parseV2(allocator, io, dir, diag);
        defer v2.deinit(allocator);
        try bank.adoptV2(&v2, implemented, diag);
        bank.sidecar_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, v2.sidecar.?.file }, 0);
        // The pool's descriptor: a regular file holding every record, the page cache bypassed and read-ahead off
        // (reads are whole spans).
        bank.sidecar = try bank_file.openSidecar(bank.sidecar_path, bank.sidecar_size, .at_least, diag);
        try bank.checkV1(io, dir, v2.sidecar.?.file, diag);
        return bank;
    }

    pub fn deinit(self: *Bank) void {
        if (self.sidecar.fd >= 0) self.sidecar.close();
        self.allocator.free(self.layers);
        self.allocator.free(self.digests);
        if (self.sidecar_path.len > 0) self.allocator.free(self.sidecar_path);
        self.* = undefined;
    }

    pub fn recordOffset(self: *const Bank, layer: u32, expert: u32) u64 {
        const l = &self.layers[layer];
        return l.base_offset + @as(u64, expert) * l.record_bytes;
    }

    pub fn spans(self: *const Bank, layer: u32, expert: u32) Spans {
        const off = self.recordOffset(layer, expert);
        const segs = &self.layers[layer].segments;
        return .{ .gu_offset = off + segs[0].offset, .down_offset = off + segs[gu_components].offset };
    }

    pub fn digest(self: *const Bank, layer: u32, expert: u32) *const Digests {
        return &self.digests[@as(usize, layer) * self.n_experts + expert];
    }

    /// First-boot rules over the parsed v2 manifest (bankv2.check_manifest),
    /// plus the dims the installed kernels implement.
    fn adoptV2(self: *Bank, v2: *const V2, implemented: Implemented, diag: ?*Diag) !void {
        const a = self.allocator;
        const q = v2.quant orelse return refuse(diag, error.ManifestSyntax, "v2: no quantization", .{});
        if (!containsStr(implemented.codebooks, q.codebook)) return refuse(diag, error.CodebookNotImplemented, "v2: codebook \"{s}\" not implemented by this build", .{q.codebook});
        if (std.mem.eql(u8, q.codebook, "mul1") and q.codebook_multiplier != mul1_multiplier)
            return refuse(diag, error.CodebookMultiplier, "v2: mul1 multiplier {d} != {d}", .{ q.codebook_multiplier, mul1_multiplier });
        const d = v2.dims orelse return refuse(diag, error.ManifestSyntax, "v2: no dims", .{});
        if (d.n_layers == 0 or d.n_layers > max_layers or d.n_experts == 0 or d.n_experts > max_experts or layerSegments(1, d.hidden, d.inter) == null)
            return refuse(diag, error.DimsInvalid, "v2: dims hidden={d} inter={d} n_experts={d} n_layers={d}", .{ d.hidden, d.inter, d.n_experts, d.n_layers });
        if (d.hidden != implemented.hidden or d.inter != implemented.inter or d.n_experts != implemented.n_experts or d.n_layers != implemented.n_layers)
            return refuse(diag, error.DimsNotImplemented, "v2: dims {d}/{d}/{d} experts/{d} layers != the implemented {d}/{d}/{d}/{d}", .{ d.hidden, d.inter, d.n_experts, d.n_layers, implemented.hidden, implemented.inter, implemented.n_experts, implemented.n_layers });
        self.hidden = d.hidden;
        self.inter = d.inter;
        self.n_experts = @intCast(d.n_experts);

        const table = v2.layers orelse return refuse(diag, error.ManifestSyntax, "v2: no layer table", .{});
        if (table.len != d.n_layers) return refuse(diag, error.LayerGeometry, "v2: {d} layer entries != dims.n_layers {d}", .{ table.len, d.n_layers });
        const ks = try a.alloc(u32, table.len);
        defer a.free(ks);
        for (table, ks, 0..) |t, *k, li| {
            if (t.layer != li) return refuse(diag, error.LayerGeometry, "v2: layer entry {d} names layer {d}", .{ li, t.layer });
            if (t.K > std.math.maxInt(u32) or !containsInt(implemented.k, @intCast(t.K))) return refuse(diag, error.KNotImplemented, "v2: layer {d} K={d} not implemented by the installed lanes", .{ li, t.K });
            k.* = @intCast(t.K);
        }
        self.layers = try a.alloc(Layer, table.len);
        const total = layerTable(ks, d.hidden, d.inter, d.n_experts, self.layers) orelse return refuse(diag, error.DimsInvalid, "v2: bank size overflows", .{});
        for (table, self.layers, 0..) |t, *want, li| {
            if (!segmentsMatch(t.segments, want, 0) or t.record_bytes != want.record_bytes or t.logical_bytes != want.logical_bytes)
                return refuse(diag, error.LayerGeometry, "v2: layer {d} segment table / record size differs from the v2 geometry for K={d}", .{ li, t.K });
            if (t.base_offset != want.base_offset) return refuse(diag, error.LayerGeometry, "v2: layer {d} base_offset {d} != {d}", .{ li, t.base_offset, want.base_offset });
        }
        const sc = v2.sidecar orelse return refuse(diag, error.ManifestSyntax, "v2: no sidecar", .{});
        if (sc.size != total) return refuse(diag, error.SidecarGeometry, "v2: sidecar.size {d} != {d}", .{ sc.size, total });
        if (sc.alignment != record_alignment) return refuse(diag, error.SidecarGeometry, "v2: sidecar alignment {d} != {d}", .{ sc.alignment, record_alignment });
        if (!bank_file.plainName(sc.file)) return refuse(diag, error.SidecarGeometry, "v2: sidecar file \"{s}\" is not a plain file name", .{sc.file});
        self.sidecar_size = total;

        const n_rec = table.len * d.n_experts;
        if (v2.records.items.len != n_rec) return refuse(diag, error.RecordCount, "v2: {d} records != {d} x {d}", .{ v2.records.items.len, table.len, d.n_experts });
        self.digests = try a.alloc(Digests, n_rec);
        const seen = try a.alloc(bool, n_rec);
        defer a.free(seen);
        @memset(seen, false);
        for (v2.records.items) |r| {
            if (r.layer >= table.len or r.expert >= d.n_experts) return refuse(diag, error.RecordGeometry, "v2: record ({d},{d}) out of range", .{ r.layer, r.expert });
            const i: usize = @intCast(r.layer * d.n_experts + r.expert);
            if (seen[i]) return refuse(diag, error.RecordDuplicate, "v2: record ({d},{d}) duplicated", .{ r.layer, r.expert });
            seen[i] = true;
            const l = &self.layers[@intCast(r.layer)];
            if (r.sidecar_offset != l.base_offset + r.expert * l.record_bytes or r.record_bytes != l.record_bytes or r.logical_bytes != l.logical_bytes)
                return refuse(diag, error.RecordGeometry, "v2: record ({d},{d}) offset/size differs from its layer geometry", .{ r.layer, r.expert });
            if (r.sidecar_offset % record_alignment != 0) return refuse(diag, error.RecordGeometry, "v2: record ({d},{d}) misaligned", .{ r.layer, r.expert });
            self.digests[i].padded = r.sha256 orelse return refuse(diag, error.RecordSha256, "v2: record ({d},{d}) has no sha256", .{ r.layer, r.expert });
        }
        if (v2.all_pass != true) return refuse(diag, error.ParityNotPassed, "v2: parity.all_pass is not true", .{});
    }

    /// The runtime (v1) manifest must describe exactly the v2 records:
    /// same offsets, logical sizes and segment table, contiguous, in order.
    fn checkV1(self: *Bank, io: std.Io, dir: []const u8, sidecar_file: []const u8, diag: ?*Diag) !void {
        const a = self.allocator;
        var src = try ManifestSource.open(a, io, dir, "expert-manifest.json", diag);
        defer src.deinit();
        var meta = std.heap.ArenaAllocator.init(a);
        defer meta.deinit();
        var rec_arena = std.heap.ArenaAllocator.init(a);
        defer rec_arena.deinit();
        const seen = try a.alloc(bool, self.digests.len);
        defer a.free(seen);
        @memset(seen, false);
        var count: usize = 0;
        var have: std.EnumSet(V1Key) = .empty;
        var sidecar: ?SidecarJson = null;

        try src.expect(.object_begin);
        while (try src.key(V1Key)) |k| {
            if (k == .unknown) {
                try src.skip();
                continue;
            }
            if (have.contains(k)) return refuse(diag, error.ManifestSyntax, "{s}: duplicate key \"{t}\"", .{ src.name, k });
            have.insert(k);
            switch (k) {
                .format => {
                    const f = try src.parse([]const u8, meta.allocator());
                    if (!std.mem.eql(u8, f, format_v1)) return refuse(diag, error.ManifestFormat, "{s}: format \"{s}\" is not {s}", .{ src.name, f, format_v1 });
                },
                .quantization => {
                    const qm = try src.parse(struct { mode: []const u8 }, meta.allocator());
                    if (!std.mem.eql(u8, qm.mode, runtime_mode)) return refuse(diag, error.RuntimeModeNotExl3, "{s}: quantization.mode \"{s}\" is not {s}", .{ src.name, qm.mode, runtime_mode });
                },
                .sidecar => sidecar = try src.parse(SidecarJson, meta.allocator()),
                .records => {
                    try src.expect(.array_begin);
                    while (try src.peek() != .array_end) {
                        _ = rec_arena.reset(.retain_capacity);
                        const r = try src.parse(V1RecordJson, rec_arena.allocator());
                        count += 1;
                        if (count > self.digests.len) return refuse(diag, error.RecordCount, "{s}: more than {d} records", .{ src.name, self.digests.len });
                        try self.checkV1Record(r, sidecar_file, seen, diag);
                    }
                    _ = try src.next();
                },
                .unknown => unreachable,
            }
        }
        if (try src.next() != .end_of_document) return refuse(diag, error.ManifestSyntax, "{s}: trailing data", .{src.name});
        if (!have.contains(.format) or !have.contains(.quantization)) return refuse(diag, error.ManifestSyntax, "{s}: no format / quantization", .{src.name});
        if (count != self.digests.len) return refuse(diag, error.RecordCount, "{s}: {d} records != {d}", .{ src.name, count, self.digests.len });
        const sc = sidecar orelse return refuse(diag, error.ManifestSyntax, "{s}: no sidecar", .{src.name});
        if (!std.mem.eql(u8, sc.file, sidecar_file) or sc.size != self.sidecar_size or sc.alignment != record_alignment)
            return refuse(diag, error.SidecarGeometry, "{s}: sidecar {s} ({d} B, align {d}) differs from the v2 manifest", .{ src.name, sc.file, sc.size, sc.alignment });
    }

    fn checkV1Record(self: *Bank, r: V1RecordJson, sidecar_file: []const u8, seen: []bool, diag: ?*Diag) !void {
        if (r.layer >= self.layers.len or r.expert >= self.n_experts) return refuse(diag, error.RuntimeRecordMismatch, "v1: record ({d},{d}) is not in the v2 bank", .{ r.layer, r.expert });
        const layer: u32 = @intCast(r.layer);
        const expert: u32 = @intCast(r.expert);
        const i = @as(usize, layer) * self.n_experts + expert;
        if (seen[i]) return refuse(diag, error.RecordDuplicate, "v1: record ({d},{d}) duplicated", .{ layer, expert });
        seen[i] = true;
        const l = &self.layers[layer];
        const off = self.recordOffset(layer, expert);
        if (r.sidecar_offset != off or r.logical_bytes != l.logical_bytes or r.sidecar_length != l.logical_bytes)
            return refuse(diag, error.RuntimeRecordMismatch, "v1: record ({d},{d}) offset/length differs from the v2 manifest", .{ layer, expert });
        if (r.segments.len != n_components) return refuse(diag, error.RuntimeRecordMismatch, "v1: record ({d},{d}) has {d} segments", .{ layer, expert, r.segments.len });
        for (r.segments) |sg| if (!std.mem.eql(u8, sg.shard, sidecar_file))
            return refuse(diag, error.RuntimeRecordMismatch, "v1: record ({d},{d}) reads shard \"{s}\"", .{ layer, expert, sg.shard });
        // Absolute offsets equal to base + the v2 table = contiguous, in record order.
        if (!segmentsMatch(r.segments, l, off)) return refuse(diag, error.RuntimeRecordMismatch, "v1: record ({d},{d}) segments differ from the v2 layer table (order, offsets, lengths, dtype, shape)", .{ layer, expert });
        self.digests[i].logical = r.sha256 orelse return refuse(diag, error.RecordSha256, "v1: record ({d},{d}) has no sha256", .{ layer, expert });
    }
};

fn containsStr(set: []const []const u8, s: []const u8) bool {
    for (set) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

fn containsInt(set: []const u32, v: u32) bool {
    return std.mem.indexOfScalar(u32, set, v) != null;
}

/// Manifest segments (v2: record-relative, base 0; v1: absolute, base =
/// record offset) against the layer table, in record order.
fn segmentsMatch(segs: anytype, l: *const Layer, base: u64) bool {
    if (segs.len != n_components) return false;
    for (segs, l.segments, 0..) |m, want, c| {
        if (!std.mem.eql(u8, m.component, Component.name(@enumFromInt(c)))) return false;
        if (!std.mem.eql(u8, m.dtype, @tagName(want.dtype))) return false;
        if (m.offset != base + want.offset or m.length != want.length) return false;
        if (!std.mem.eql(u64, m.shape, want.shape[0..want.rank])) return false;
    }
    return true;
}

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

// ── Manifest parsing: token-streamed, one record in memory at a time ──

/// A lowercase-hex sha256 decodes; anything else is "no sha256".
const Sha256Hex = struct {
    fn parse(s: []const u8) ?[32]u8 {
        if (s.len != 64) return null;
        for (s) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
        var out: [32]u8 = undefined;
        _ = std.fmt.hexToBytes(&out, s) catch return null;
        return out;
    }
};

const SegJson = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
const V1SegJson = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64, shard: []const u8 };
const SidecarJson = struct { file: []const u8, alignment: u64, size: u64 };

const V1RecordJson = struct {
    layer: u64,
    expert: u64,
    logical_bytes: u64,
    segments: []const V1SegJson,
    sha256: ?[32]u8,
    sidecar_offset: u64,
    sidecar_length: u64,

    pub fn jsonParse(a: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !V1RecordJson {
        const raw = try std.json.innerParse(struct {
            layer: u64,
            expert: u64,
            logical_bytes: u64,
            segments: []const V1SegJson,
            sha256: []const u8 = "",
            sidecar_offset: u64,
            sidecar_length: u64,
        }, a, source, options);
        return .{ .layer = raw.layer, .expert = raw.expert, .logical_bytes = raw.logical_bytes, .segments = raw.segments, .sha256 = Sha256Hex.parse(raw.sha256), .sidecar_offset = raw.sidecar_offset, .sidecar_length = raw.sidecar_length };
    }
};

const V1Key = enum { format, quantization, records, sidecar, unknown };
const V2Key = enum { format, quantization, dims, sidecar, layers, records, parity, unknown };

const V2 = struct {
    quant: ?struct { codebook: []const u8, codebook_multiplier: u64 } = null,
    dims: ?struct { hidden: u64, inter: u64, n_experts: u64, n_layers: u64 } = null,
    sidecar: ?SidecarJson = null,
    layers: ?[]const struct { layer: u64, K: u64, record_bytes: u64, logical_bytes: u64, base_offset: u64, segments: []const SegJson } = null,
    records: std.ArrayList(Rec) = .empty,
    all_pass: ?bool = null,
    meta: std.heap.ArenaAllocator,

    const Rec = struct { layer: u64, expert: u64, sidecar_offset: u64, record_bytes: u64, logical_bytes: u64, sha256: ?[32]u8 };

    fn deinit(self: *V2, a: std.mem.Allocator) void {
        self.records.deinit(a);
        self.meta.deinit();
    }
};

fn parseV2(a: std.mem.Allocator, io: std.Io, dir: []const u8, diag: ?*Diag) !V2 {
    var src = try ManifestSource.open(a, io, dir, "expert-manifest-v2.json", diag);
    defer src.deinit();
    var v2: V2 = .{ .meta = std.heap.ArenaAllocator.init(a) };
    errdefer v2.deinit(a);
    const meta = v2.meta.allocator();
    var rec_arena = std.heap.ArenaAllocator.init(a);
    defer rec_arena.deinit();
    var have: std.EnumSet(V2Key) = .empty;

    try src.expect(.object_begin);
    while (try src.key(V2Key)) |k| {
        if (k == .unknown) {
            try src.skip();
            continue;
        }
        if (have.contains(k)) return refuse(diag, error.ManifestSyntax, "{s}: duplicate key \"{t}\"", .{ src.name, k });
        have.insert(k);
        switch (k) {
            .format => {
                const f = try src.parse([]const u8, meta);
                if (!std.mem.eql(u8, f, format_v2)) return refuse(diag, error.ManifestFormat, "{s}: format \"{s}\" is not {s}", .{ src.name, f, format_v2 });
            },
            .quantization => v2.quant = try src.parse(@TypeOf(v2.quant.?), meta),
            .dims => v2.dims = try src.parse(@TypeOf(v2.dims.?), meta),
            .sidecar => v2.sidecar = try src.parse(SidecarJson, meta),
            .layers => {
                v2.layers = try src.parse(@TypeOf(v2.layers.?), meta);
                if (v2.layers.?.len > max_layers) return refuse(diag, error.LayerGeometry, "{s}: {d} layers", .{ src.name, v2.layers.?.len });
            },
            .records => {
                try src.expect(.array_begin);
                while (try src.peek() != .array_end) {
                    _ = rec_arena.reset(.retain_capacity);
                    const r = try src.parse(struct { layer: u64, expert: u64, sidecar_offset: u64, record_bytes: u64, logical_bytes: u64, sha256: []const u8 = "" }, rec_arena.allocator());
                    if (v2.records.items.len == max_records) return refuse(diag, error.RecordCount, "{s}: more than {d} records", .{ src.name, max_records });
                    try v2.records.append(a, .{ .layer = r.layer, .expert = r.expert, .sidecar_offset = r.sidecar_offset, .record_bytes = r.record_bytes, .logical_bytes = r.logical_bytes, .sha256 = Sha256Hex.parse(r.sha256) });
                }
                _ = try src.next();
            },
            .parity => v2.all_pass = (try src.parse(struct { all_pass: bool }, meta)).all_pass,
            .unknown => unreachable,
        }
    }
    if (try src.next() != .end_of_document) return refuse(diag, error.ManifestSyntax, "{s}: trailing data", .{src.name});
    if (!have.contains(.format)) return refuse(diag, error.ManifestFormat, "{s}: no format", .{src.name});
    return v2;
}

/// One manifest file as a JSON token stream: a fixed read buffer and one
/// record's allocations at a time, whatever the file size.
const ManifestSource = struct {
    file: std.Io.File,
    io: std.Io,
    buf: []u8,
    reader: std.Io.File.Reader,
    json: std.json.Reader,
    diagnostics: std.json.Diagnostics = .{},
    name: []const u8,
    diag: ?*Diag,
    a: std.mem.Allocator,

    const parse_options: std.json.ParseOptions = .{ .ignore_unknown_fields = true, .max_value_len = 1 << 16, .allocate = .alloc_always };

    fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, diag: ?*Diag) !*ManifestSource {
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
        defer a.free(path);
        const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch |e| switch (e) {
            error.FileNotFound => return refuse(diag, error.ManifestMissing, "{s}: not found", .{name}),
            else => |other| return other,
        };
        errdefer file.close(io);
        // A bank read like the records': streamed once, past the page cache.
        io_util.noCache(file.handle, .{ .read_ahead = true }) catch return refuse(diag, error.ManifestOpen, "{s}: F_NOCACHE refused", .{name});
        const self = try a.create(ManifestSource);
        errdefer a.destroy(self);
        const buf = try a.alloc(u8, 1 << 16);
        self.* = .{ .file = file, .io = io, .buf = buf, .reader = file.reader(io, buf), .json = undefined, .name = name, .diag = diag, .a = a };
        self.json = std.json.Reader.init(a, &self.reader.interface);
        self.json.enableDiagnostics(&self.diagnostics);
        return self;
    }

    fn deinit(self: *ManifestSource) void {
        self.json.deinit();
        self.file.close(self.io);
        self.a.free(self.buf);
        self.a.destroy(self);
    }

    fn syntax(self: *ManifestSource, err: anyerror) Refusal {
        return refuse(self.diag, error.ManifestSyntax, "{s}: {s} at byte {d}", .{ self.name, @errorName(err), self.diagnostics.getByteOffset() });
    }

    fn parse(self: *ManifestSource, comptime T: type, a: std.mem.Allocator) !T {
        return std.json.innerParse(T, a, &self.json, parse_options) catch |e| switch (e) {
            error.OutOfMemory, error.ReadFailed => |x| return x,
            else => return self.syntax(e),
        };
    }

    fn peek(self: *ManifestSource) !std.json.TokenType {
        return self.json.peekNextTokenType() catch |e| switch (e) {
            error.ReadFailed => |x| return x,
            else => return self.syntax(e),
        };
    }

    fn next(self: *ManifestSource) !std.json.Token {
        return self.json.next() catch |e| switch (e) {
            error.OutOfMemory, error.ReadFailed => |x| return x,
            else => return self.syntax(e),
        };
    }

    fn expect(self: *ManifestSource, want: std.json.TokenType) !void {
        if (try self.peek() != want) return self.syntax(error.UnexpectedToken);
        _ = try self.next();
    }

    fn skip(self: *ManifestSource) !void {
        self.json.skipValue() catch |e| switch (e) {
            error.OutOfMemory, error.ReadFailed => |x| return x,
            else => return self.syntax(e),
        };
    }

    /// The next object key as `K` (`.unknown` for any other), or null at `}`.
    fn key(self: *ManifestSource, comptime K: type) !?K {
        const tok = self.json.nextAlloc(self.a, .alloc_if_needed) catch |e| switch (e) {
            error.OutOfMemory, error.ReadFailed => |x| return x,
            else => return self.syntax(e),
        };
        switch (tok) {
            .object_end => return null,
            .string => |s| return std.meta.stringToEnum(K, s) orelse .unknown,
            .allocated_string => |s| {
                defer self.a.free(s);
                return std.meta.stringToEnum(K, s) orelse .unknown;
            },
            else => return self.syntax(error.UnexpectedToken),
        }
    }
};

// ── C2: the bank's description for the quants' claims ──

const quant = @import("../sdk_ext.zig").quant;

/// The bank's description (`quant.BankPeek`) from its v2 manifest, for the load path's quant
/// `claims`: the `quantization` object whole (each quant reads its own fields), the dims, and
/// every layer's K and segment table. Read on its own, before the quant the bank's arch binds is
/// accepted and before `Bank.open`; the records are skipped.
pub const Peek = struct {
    arena: std.heap.ArenaAllocator,
    view: quant.BankPeek,

    pub fn deinit(self: *Peek) void {
        self.arena.deinit();
    }

    fn adopt(p: *Peek, j: PeekJson, diag: ?*Diag) !void {
        const aa = p.arena.allocator();
        if (!std.mem.eql(u8, j.format, format_v2)) return refuse(diag, error.ManifestFormat, "v2 peek: format \"{s}\" is not {s}", .{ j.format, format_v2 });
        if (j.layers.len > max_layers) return refuse(diag, error.LayerGeometry, "v2 peek: {d} layers", .{j.layers.len});
        const layers = try aa.alloc(quant.LayerPeek, j.layers.len);
        for (j.layers, layers, 0..) |l, *o, li| {
            if (l.layer != li) return refuse(diag, error.LayerGeometry, "v2 peek: layer entry {d} names layer {d}", .{ li, l.layer });
            if (l.K > std.math.maxInt(u32)) return refuse(diag, error.KNotImplemented, "v2 peek: layer {d} K={d}", .{ li, l.K });
            const segs = try aa.alloc(quant.Segment, l.segments.len);
            for (l.segments, segs) |s, *d| d.* = .{ .name = s.component, .dtype = s.dtype, .shape = s.shape };
            o.* = .{ .bits = @intCast(l.K), .segments = segs };
        }
        p.view = .{ .quantization = j.quantization, .hidden = j.dims.hidden, .inter = j.dims.inter, .n_experts = j.dims.n_experts, .n_layers = j.dims.n_layers, .layers = layers };
    }
};

const PeekJson = struct {
    format: []const u8,
    quantization: std.json.Value,
    dims: struct { hidden: u64, inter: u64, n_experts: u64, n_layers: u64 },
    layers: []const struct { layer: u64, K: u64, segments: []const SegJson },
};

/// `Peek` of a v2 manifest's text.
pub fn peekText(a: std.mem.Allocator, text: []const u8, diag: ?*Diag) !Peek {
    var p: Peek = .{ .arena = .init(a), .view = undefined };
    errdefer p.arena.deinit();
    const j = std.json.parseFromSliceLeaky(PeekJson, p.arena.allocator(), text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |e| switch (e) {
        error.OutOfMemory => |x| return x,
        else => return refuse(diag, error.ManifestSyntax, "v2 peek: {s}", .{@errorName(e)}),
    };
    try p.adopt(j, diag);
    return p;
}

/// `Peek` of the bank at `dir`: its expert-manifest-v2.json streamed past the page cache, as `Bank.open` reads it,
/// the records skipped.
pub fn peek(a: std.mem.Allocator, io: std.Io, dir: []const u8, diag: ?*Diag) !Peek {
    if (!std.fs.path.isAbsolute(dir)) return refuse(diag, error.BankDirNotAbsolute, "bank dir \"{s}\" is not an absolute path", .{dir});
    var src = try ManifestSource.open(a, io, dir, "expert-manifest-v2.json", diag);
    defer src.deinit();
    var p: Peek = .{ .arena = .init(a), .view = undefined };
    errdefer p.arena.deinit();
    const aa = p.arena.allocator();
    const Key = enum { format, quantization, dims, layers, unknown };
    var format: ?[]const u8 = null;
    var quantization: ?std.json.Value = null;
    var dims: ?@FieldType(PeekJson, "dims") = null;
    var layers: ?@FieldType(PeekJson, "layers") = null;
    try src.expect(.object_begin);
    while (try src.key(Key)) |k| {
        switch (k) {
            .format => format = try src.parse([]const u8, aa),
            .quantization => quantization = try src.parse(std.json.Value, aa),
            .dims => dims = try src.parse(@FieldType(PeekJson, "dims"), aa),
            .layers => layers = try src.parse(@FieldType(PeekJson, "layers"), aa),
            .unknown => try src.skip(),
        }
    }
    if (format == null or quantization == null or dims == null or layers == null)
        return refuse(diag, error.ManifestSyntax, "v2 peek: no format / quantization / dims / layers", .{});
    try p.adopt(.{ .format = format.?, .quantization = quantization.?, .dims = dims.?, .layers = layers.? }, diag);
    return p;
}

// ── Tests ──

const testing = std.testing;

/// Deterministic, position-dependent bytes: every offset gets its own value.
pub fn fillPattern(buf: []u8, seed: u64) void {
    var x: u64 = seed;
    for (buf) |*b| {
        x +%= 0x9E3779B97F4A7C15;
        var z = x;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        b.* = @truncate(z ^ (z >> 31));
    }
}

fn expectSegments(l: Layer, want: []const [5]u64) !void {
    for (l.segments, want) |s, w| {
        try testing.expectEqual(w[0], s.offset);
        try testing.expectEqual(w[1], s.length);
        try testing.expectEqualSlices(u64, w[2..][0..s.rank], s.shape[0..s.rank]);
    }
}

test "dsv41 bank: geometry matches bankv2" {
    // Goldens printed by bankv2.layer_table (hidden 5120 / inter 2304 / 384 experts, and 64 / 32 / 4).
    const k3 = layerSegments(3, 5120, 2304).?;
    try testing.expectEqual(@as(u64, 13_315_584), k3.logical_bytes);
    try testing.expectEqual(@as(u64, 13_316_096), k3.record_bytes);
    try expectSegments(k3, &.{
        .{ 0, 4423680, 320, 144, 48 },      .{ 4423680, 4608, 2304, 0, 0 },  .{ 4428288, 10240, 5120, 0, 0 },
        .{ 4438528, 4423680, 320, 144, 48 }, .{ 8862208, 4608, 2304, 0, 0 },  .{ 8866816, 10240, 5120, 0, 0 },
        .{ 8877056, 4423680, 144, 320, 48 }, .{ 13300736, 10240, 5120, 0, 0 }, .{ 13310976, 4608, 2304, 0, 0 },
    });
    const k2 = layerSegments(2, 5120, 2304).?;
    try testing.expectEqual(@as(u64, 8_891_904), k2.logical_bytes);
    try testing.expectEqual(@as(u64, 8_892_416), k2.record_bytes);
    try expectSegments(k2, &.{
        .{ 0, 2949120, 320, 144, 32 },      .{ 2949120, 4608, 2304, 0, 0 }, .{ 2953728, 10240, 5120, 0, 0 },
        .{ 2963968, 2949120, 320, 144, 32 }, .{ 5913088, 4608, 2304, 0, 0 }, .{ 5917696, 10240, 5120, 0, 0 },
        .{ 5927936, 2949120, 144, 320, 32 }, .{ 8877056, 10240, 5120, 0, 0 }, .{ 8887296, 4608, 2304, 0, 0 },
    });
    var table: [40]Layer = undefined;
    try testing.expectEqual(@as(?u64, 13_641_449_472), layerTable(&.{ 3, 2, 3 }, 5120, 2304, 384, table[0..3]));
    try testing.expectEqual(@as(u64, 5_113_380_864), table[1].base_offset);
    try testing.expectEqual(@as(u64, 8_528_068_608), table[2].base_offset);
    const all3: [40]u32 = @splat(3);
    try testing.expectEqual(@as(?u64, 204_535_234_560), layerTable(&all3, 5120, 2304, 384, &table));
    try testing.expectEqual(@as(u64, 66_700_324_864), table[13].base_offset + 17 * table[13].record_bytes);

    const s3 = layerSegments(3, 64, 32).?;
    try testing.expectEqual(@as(u64, 2880), s3.logical_bytes);
    try testing.expectEqual(@as(u64, 4096), s3.record_bytes);
    try expectSegments(s3, &.{
        .{ 0, 768, 4, 2, 48 },    .{ 768, 64, 32, 0, 0 },   .{ 832, 128, 64, 0, 0 },
        .{ 960, 768, 4, 2, 48 },  .{ 1728, 64, 32, 0, 0 },  .{ 1792, 128, 64, 0, 0 },
        .{ 1920, 768, 2, 4, 48 }, .{ 2688, 128, 64, 0, 0 }, .{ 2816, 64, 32, 0, 0 },
    });
    const s2 = layerSegments(2, 64, 32).?;
    try testing.expectEqual(@as(u64, 2112), s2.logical_bytes);
    try testing.expectEqual(@as(u64, 4096), s2.record_bytes);
    try testing.expectEqual(@as(u64, 1408), s2.segments[gu_components].offset);
    try testing.expectEqual(@as(?u64, 49_152), layerTable(&.{ 3, 2, 3 }, 64, 32, 4, table[0..3]));
    try testing.expectEqual(@as(u64, 32_768), table[2].base_offset);
    try testing.expect(layerSegments(3, 5121, 2304) == null);
    try testing.expect(layerSegments(0, 5120, 2304) == null);
}

// Synthetic banks: the geometry is written out here independently of
// `layerSegments`, so a bug there cannot be mirrored by the generator.
pub const Synth = struct {
    hidden: u64 = 64,
    inter: u64 = 32,
    n_experts: u32 = 4,
    /// K per layer (at most `max_synth_layers` layers).
    k: []const u32 = &.{ 3, 3 },
    v2_format: []const u8 = "mtplx-expert-manifest-v2",
    codebook: []const u8 = "mul1",
    multiplier: u64 = 0x83DCD12D,
    drop_layer_entry: bool = false,
    layer_index_delta: u64 = 0,
    seg_len_delta: u64 = 0,
    base_delta: u64 = 0,
    sidecar_size_delta: u64 = 0,
    sidecar_alignment: u64 = 4096,
    drop_record: bool = false,
    dup_record: bool = false,
    record_out_of_range: bool = false,
    record_offset_delta: u64 = 0,
    bad_sha: bool = false,
    all_pass: bool = true,
    truncate_file: bool = false,
    no_file: bool = false,
    sidecar_symlink: bool = false,
    no_v2: bool = false,
    v2_truncated: bool = false,
    v1_format: []const u8 = "mtplx-expert-manifest-v1",
    v1_mode: []const u8 = "exl3-mul1",
    v1_offset_delta: u64 = 0,
    v1_swap: bool = false,
    v1_gap: bool = false,
    v1_sha_short: bool = false,
    v1_drop_record: bool = false,
    v1_truncated: bool = false,
    no_v1: bool = false,
    /// The F16 components (rout, rin) as small finite scales, not pattern bytes: a bank the kernels decode to finite rows.
    finite_scales: bool = false,
};

const SynthSeg = struct { name: []const u8, dtype: []const u8, shape: [3]u64, rank: usize, offset: u64, length: u64 };

fn synthSegments(k: u64, hidden: u64, inter: u64, out: *[9]SynthSeg) u64 {
    const names = [_][]const u8{
        "gate_proj.code", "gate_proj.rout", "gate_proj.rin",
        "up_proj.code",   "up_proj.rout",   "up_proj.rin",
        "down_proj.code", "down_proj.rout", "down_proj.rin",
    };
    var off: u64 = 0;
    for (0..3) |p| {
        const i: u64 = if (p == 2) inter else hidden;
        const o: u64 = if (p == 2) hidden else inter;
        const segs = [3]SynthSeg{
            .{ .name = names[p * 3], .dtype = "I16", .shape = .{ i / 16, o / 16, 16 * k }, .rank = 3, .offset = 0, .length = (i / 16) * (o / 16) * 16 * k * 2 },
            .{ .name = names[p * 3 + 1], .dtype = "F16", .shape = .{ o, 0, 0 }, .rank = 1, .offset = 0, .length = o * 2 },
            .{ .name = names[p * 3 + 2], .dtype = "F16", .shape = .{ i, 0, 0 }, .rank = 1, .offset = 0, .length = i * 2 },
        };
        for (segs, 0..) |sg, j| {
            out[p * 3 + j] = sg;
            out[p * 3 + j].offset = off;
            off += sg.length;
        }
    }
    return off;
}

fn printShape(j: *std.ArrayList(u8), a: std.mem.Allocator, sg: SynthSeg) !void {
    try j.appendSlice(a, "[");
    for (sg.shape[0..sg.rank], 0..) |d, n| try j.print(a, "{s}{d}", .{ if (n == 0) "" else ",", d });
    try j.appendSlice(a, "]");
}

const max_synth_layers = 8;

/// Writes a bank of `s.k.len` layers (manifests + experts.bin) into `tmp`;
/// returns the experts.bin image (caller frees).
pub fn writeSynth(a: std.mem.Allocator, tmp: *std.testing.TmpDir, s: Synth) ![]u8 {
    const io = std.testing.io;
    const nl = s.k.len;
    std.debug.assert(nl >= 2 and nl <= max_synth_layers);
    var segs: [max_synth_layers][9]SynthSeg = undefined;
    var logical: [max_synth_layers]u64 = undefined;
    var rb: [max_synth_layers]u64 = undefined;
    var base: [max_synth_layers]u64 = undefined;
    var total: u64 = 0;
    for (0..nl) |l| {
        logical[l] = synthSegments(s.k[l], s.hidden, s.inter, &segs[l]);
        rb[l] = (logical[l] + 4095) / 4096 * 4096;
        base[l] = total;
        total += s.n_experts * rb[l];
    }
    const bin = try a.alloc(u8, total);
    errdefer a.free(bin);
    @memset(bin, 0);
    const n_rec = nl * s.n_experts;
    const lsha = try a.alloc([64]u8, n_rec);
    defer a.free(lsha);
    const psha = try a.alloc([64]u8, n_rec);
    defer a.free(psha);
    for (0..nl) |l| for (0..s.n_experts) |e| {
        const off = base[l] + e * rb[l];
        fillPattern(bin[off .. off + logical[l]], 1 + l * 1000 + e);
        if (s.finite_scales) for (segs[l]) |sg| if (std.mem.eql(u8, sg.dtype, "F16")) for (0..sg.length / 2) |i| {
            const v: f16 = @floatFromInt((i + e) % 7 + 1);
            std.mem.writeInt(u16, bin[off + sg.offset + 2 * i ..][0..2], @bitCast(v / 64), .little);
        };
        lsha[l * s.n_experts + e] = kset.shaHex(bin[off .. off + logical[l]]);
        psha[l * s.n_experts + e] = kset.shaHex(bin[off .. off + rb[l]]);
    };

    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    // v2
    try j.print(a, "{{\"format\":\"{s}\",\"model_key\":\"synthetic\",\"quantization\":{{\"mode\":\"exl3\",\"codebook\":\"{s}\",\"codebook_multiplier\":{d},\"tile\":{{\"size\":16}}}},", .{ s.v2_format, s.codebook, s.multiplier });
    try j.print(a, "\"dims\":{{\"hidden\":{d},\"inter\":{d},\"n_experts\":{d},\"n_layers\":{d}}},\"sidecar\":{{\"file\":\"experts.bin\",\"alignment\":{d},\"size\":{d}}},\"layers\":[", .{ s.hidden, s.inter, s.n_experts, nl, s.sidecar_alignment, total + s.sidecar_size_delta });
    for (0..nl) |l| {
        if (s.drop_layer_entry and l == 1) continue;
        const index = l + (if (l == 1) s.layer_index_delta else 0);
        try j.print(a, "{s}{{\"layer\":{d},\"K\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"base_offset\":{d},\"segments\":[", .{ if (l == 0) "" else ",", index, s.k[l], rb[l], logical[l], base[l] + (if (l == 1) s.base_delta else 0) });
        for (segs[l], 0..) |sg, n| {
            const len = sg.length + (if (l == 1 and n == 5) s.seg_len_delta else 0);
            try j.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"shape\":", .{ if (n == 0) "" else ",", sg.name, sg.dtype });
            try printShape(&j, a, sg);
            try j.print(a, ",\"offset\":{d},\"length\":{d}}}", .{ sg.offset, len });
        }
        try j.appendSlice(a, "]}");
    }
    try j.appendSlice(a, "],\"records\":[");
    var first = true;
    for (0..nl) |l| for (0..s.n_experts) |e| {
        if (s.drop_record and l == 1 and e == 3) continue;
        const ee = if (s.dup_record and l == 1 and e == 3) 2 else e;
        var off = base[l] + ee * rb[l];
        if (l == 1 and e == 2) off += s.record_offset_delta;
        const named = if (s.record_out_of_range and l == 1 and e == 3) s.n_experts else ee;
        const sha: []const u8 = if (s.bad_sha and l == 0 and e == 1) "abc" else &psha[l * s.n_experts + ee];
        try j.print(a, "{s}{{\"layer\":{d},\"expert\":{d},\"sidecar_offset\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"sha256\":\"{s}\",\"stats\":{{\"max_abs_rin\":[0.5,0.25,0.125],\"rout_zeros\":[0,0,0]}}}}", .{ if (first) "" else ",", l, named, off, rb[l], logical[l], sha });
        first = false;
    };
    try j.print(a, "],\"artifact\":{{\"record_count\":{d},\"routed_expert_bytes\":{d}}},\"parity\":{{\"receipts\":\"receipts\",\"all_pass\":{}}}}}", .{ n_rec, total, s.all_pass });
    if (!s.no_v2) {
        const v2 = if (s.v2_truncated) j.items[0 .. j.items.len / 2] else j.items;
        try tmp.dir.writeFile(io, .{ .sub_path = "expert-manifest-v2.json", .data = v2 });
    }

    // v1 (runtime): absolute segment offsets, logical-bytes digest
    j.clearRetainingCapacity();
    try j.print(a, "{{\"artifact\":{{\"record_count\":{d}}},\"format\":\"{s}\",\"manifest_sha256\":\"{s}\",\"model_key\":\"synthetic\",\"quantization\":{{\"bits\":3,\"group_size\":32,\"mode\":\"{s}\"}},\"records\":[", .{ n_rec, s.v1_format, lsha[0], s.v1_mode });
    first = true;
    for (0..nl) |l| for (0..s.n_experts) |e| {
        if (s.v1_drop_record and l == 1 and e == 0) continue;
        const off = base[l] + e * rb[l];
        try j.print(a, "{s}{{\"expert\":{d},\"layer\":{d},\"logical_bytes\":{d},\"segments\":[", .{ if (first) "" else ",", e, l, logical[l] });
        first = false;
        var order = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
        if (s.v1_swap and l == 0 and e == 2) std.mem.swap(usize, &order[1], &order[2]);
        for (order, 0..) |n, pos| {
            const sg = segs[l][n];
            const seg_off = off + sg.offset + (if (s.v1_gap and l == 0 and e == 3 and n == 6) @as(u64, 2) else 0);
            try j.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"length\":{d},\"offset\":{d},\"shape\":", .{ if (pos == 0) "" else ",", sg.name, sg.dtype, sg.length, seg_off });
            try printShape(&j, a, sg);
            try j.print(a, ",\"shard\":\"experts.bin\",\"tensor\":\"layers.{d}.ffn.experts.{d}\"}}", .{ l, e });
        }
        const sha: []const u8 = if (s.v1_sha_short and l == 0 and e == 0) "abcd" else &lsha[l * s.n_experts + e];
        const v1_off = off + (if (l == 1 and e == 1) s.v1_offset_delta else 0);
        try j.print(a, "],\"sha256\":\"{s}\",\"sidecar_length\":{d},\"sidecar_offset\":{d}}}", .{ sha, logical[l], v1_off });
    };
    try j.print(a, "],\"resident_tensors\":[{{\"tensor\":\"embed\",\"shard\":\"model.safetensors\",\"offset\":0,\"length\":4,\"dtype\":\"F16\",\"shape\":[2]}}],\"shards\":[{{\"name\":\"experts.bin\",\"size\":{d}}}],\"sidecar\":{{\"alignment\":4096,\"file\":\"experts.bin\",\"sha256\":\"{s}\",\"size\":{d}}},\"source_repo\":\"synthetic\"}}", .{ total, psha[0], total });
    if (!s.no_v1) {
        // Truncated: the stream ends between two records.
        const v1 = if (s.v1_truncated) j.items[0 .. std.mem.indexOf(u8, j.items, "},{\"expert\"").? + 1] else j.items;
        try tmp.dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = v1 });
    }

    if (!s.no_file) {
        const n = if (s.truncate_file) total - 4096 else total;
        try tmp.dir.writeFile(io, .{ .sub_path = if (s.sidecar_symlink) "experts.real" else "experts.bin", .data = bin[0..n] });
        if (s.sidecar_symlink) try tmp.dir.symLink(io, "experts.real", "experts.bin", .{});
    }
    return bin;
}

const implemented_synth: Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = 4, .n_layers = 2 };

pub const tmpRoot = @import("sdk").testing.tmpRoot;

/// A synthetic bank (`writeSynth`) opened from a temporary directory (tests).
pub const SynthBank = struct {
    tmp: std.testing.TmpDir,
    image: []u8,
    bank: Bank,

    pub fn open(n_experts: u32) !SynthBank {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const image = try writeSynth(testing.allocator, &tmp, .{ .n_experts = n_experts });
        errdefer testing.allocator.free(image);
        var rbuf: [512]u8 = undefined;
        const implemented: Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = n_experts, .n_layers = 2 };
        const bank = try Bank.open(testing.allocator, std.testing.io, try tmpRoot(&tmp, &rbuf), implemented, null);
        return .{ .tmp = tmp, .image = image, .bank = bank };
    }

    pub fn close(self: *SynthBank) void {
        self.bank.deinit();
        testing.allocator.free(self.image);
        self.tmp.cleanup();
    }
};

test "dsv41 bank: a clean synthetic bank opens with offsets, spans and digests from its manifests" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bin = try writeSynth(testing.allocator, &tmp, .{});
    defer testing.allocator.free(bin);
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    var bank = Bank.open(testing.allocator, std.testing.io, try tmpRoot(&tmp, &rbuf), implemented_synth, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    try testing.expectEqual(@as(usize, 2), bank.layers.len);
    try testing.expectEqual(@as(u32, 4), bank.n_experts);
    try testing.expectEqual(@as(u64, 4 * 4096), bank.layers[1].base_offset);
    try testing.expectEqual(@as(u64, 6 * 4096), bank.recordOffset(1, 2));
    try testing.expectEqual(Spans{ .gu_offset = 6 * 4096, .down_offset = 6 * 4096 + 1920 }, bank.spans(1, 2));
    try testing.expectEqual(@as(u64, bin.len), bank.sidecar_size);
    try testing.expectEqual(@as(u64, bin.len), bank.sidecar.size);
    try testing.expect(bank.sidecar.fd >= 0);
    // Both digests of every record are the manifests' (checked against the image here).
    for (0..2) |l| for (0..4) |e| {
        const off = bank.recordOffset(@intCast(l), @intCast(e));
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bin[off..][0..4096], &d, .{});
        try testing.expectEqualSlices(u8, &d, &bank.digest(@intCast(l), @intCast(e)).padded);
        std.crypto.hash.sha2.Sha256.hash(bin[off..][0..2880], &d, .{});
        try testing.expectEqualSlices(u8, &d, &bank.digest(@intCast(l), @intCast(e)).logical);
    };
}

test "dsv41 bank: a K=2 layer opens only when K=2 is implemented" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bin = try writeSynth(testing.allocator, &tmp, .{ .k = &.{ 3, 2 } });
    defer testing.allocator.free(bin);
    var rbuf: [512]u8 = undefined;
    const root = try tmpRoot(&tmp, &rbuf);
    try testing.expectError(error.KNotImplemented, Bank.open(testing.allocator, std.testing.io, root, implemented_synth, null));
    var k23 = implemented_synth;
    k23.k = &.{ 2, 3 };
    var bank = try Bank.open(testing.allocator, std.testing.io, root, k23, null);
    defer bank.deinit();
    try testing.expectEqual(@as(u32, 2), bank.layers[1].k);
    try testing.expectEqual(@as(u64, 2112), bank.layers[1].logical_bytes);
    try testing.expectEqual(Spans{ .gu_offset = 4 * 4096, .down_offset = 4 * 4096 + 1408 }, bank.spans(1, 0));
}

test "dsv41 bank: a relative or empty bank dir is refused, never opened" {
    var diag: Diag = .{};
    try testing.expectError(error.BankDirNotAbsolute, Bank.open(testing.allocator, std.testing.io, "models/bank", dsv41, &diag));
    try testing.expectError(error.BankDirNotAbsolute, Bank.open(testing.allocator, std.testing.io, "", dsv41, &diag));
    try testing.expect(diag.message().len > 0);
}

test "dsv41 bank: every bankv2 refusal refuses, by name" {
    const Case = struct { s: Synth, err: anyerror, impl: Implemented = implemented_synth };
    const cases = [_]Case{
        .{ .s = .{ .no_v2 = true }, .err = error.ManifestMissing },
        .{ .s = .{ .v2_truncated = true }, .err = error.ManifestSyntax },
        .{ .s = .{ .v2_format = "mtplx-expert-manifest-v3" }, .err = error.ManifestFormat },
        .{ .s = .{ .codebook = "mcg" }, .err = error.CodebookNotImplemented },
        .{ .s = .{ .multiplier = 0x83DCD12E }, .err = error.CodebookMultiplier },
        .{ .s = .{ .hidden = 72 }, .err = error.DimsInvalid },
        .{ .s = .{}, .err = error.DimsNotImplemented, .impl = dsv41 },
        .{ .s = .{ .drop_layer_entry = true }, .err = error.LayerGeometry },
        .{ .s = .{ .layer_index_delta = 4 }, .err = error.LayerGeometry },
        .{ .s = .{ .k = &.{ 3, 2 } }, .err = error.KNotImplemented },
        .{ .s = .{ .seg_len_delta = 2 }, .err = error.LayerGeometry },
        .{ .s = .{ .base_delta = 4096 }, .err = error.LayerGeometry },
        .{ .s = .{ .sidecar_size_delta = 4096 }, .err = error.SidecarGeometry },
        .{ .s = .{ .sidecar_alignment = 8192 }, .err = error.SidecarGeometry },
        .{ .s = .{ .drop_record = true }, .err = error.RecordCount },
        .{ .s = .{ .record_out_of_range = true }, .err = error.RecordGeometry },
        .{ .s = .{ .dup_record = true }, .err = error.RecordDuplicate },
        .{ .s = .{ .record_offset_delta = 4096 }, .err = error.RecordGeometry },
        .{ .s = .{ .bad_sha = true }, .err = error.RecordSha256 },
        .{ .s = .{ .all_pass = false }, .err = error.ParityNotPassed },
        .{ .s = .{ .no_file = true }, .err = error.SidecarMissing },
        .{ .s = .{ .truncate_file = true }, .err = error.SidecarGeometry },
        .{ .s = .{ .sidecar_symlink = true }, .err = error.SidecarOpen },
        .{ .s = .{ .no_v1 = true }, .err = error.ManifestMissing },
        .{ .s = .{ .v1_format = "mtplx-expert-manifest-v2" }, .err = error.ManifestFormat },
        .{ .s = .{ .v1_mode = "tcq3" }, .err = error.RuntimeModeNotExl3 },
        .{ .s = .{ .v1_offset_delta = 4096 }, .err = error.RuntimeRecordMismatch },
        .{ .s = .{ .v1_swap = true }, .err = error.RuntimeRecordMismatch },
        .{ .s = .{ .v1_gap = true }, .err = error.RuntimeRecordMismatch },
        .{ .s = .{ .v1_sha_short = true }, .err = error.RecordSha256 },
        .{ .s = .{ .v1_drop_record = true }, .err = error.RecordCount },
        .{ .s = .{ .v1_truncated = true }, .err = error.ManifestSyntax },
    };
    for (cases, 0..) |c, i| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        testing.allocator.free(try writeSynth(testing.allocator, &tmp, c.s));
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        if (Bank.open(testing.allocator, std.testing.io, try tmpRoot(&tmp, &rbuf), c.impl, &diag)) |opened| {
            var b = opened;
            b.deinit();
            std.debug.print("case {d}: opened, wanted {s}\n", .{ i, @errorName(c.err) });
            return error.TestUnexpectedResult;
        } else |e| {
            testing.expectEqual(c.err, e) catch |x| {
                std.debug.print("case {d}: {s}\n", .{ i, diag.message() });
                return x;
            };
            try testing.expect(diag.message().len > 0);
        }
    }
}

// DSV41_BANK=<bank dir> [DSV41_PHASE0_FIXTURE=<json from the reference runtime's dump_phase0_reader_fixture.py>]
test "dsv41 bank: the real bank opens (the 3.0 bpw bank or the 1017 2.9 bpw bank)" {
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse return error.SkipZigTest);
    var diag: Diag = .{};
    var bank = Bank.open(testing.allocator, std.testing.io, dir, dsv41, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    try testing.expectEqual(@as(usize, 40 * 384), bank.digests.len);
    try testing.expectEqual(bank.sidecar_size, bank.sidecar.size);
    // the 3.0 bpw bank (K = 3 everywhere): 204,535,234,560 bytes; the 1017 2.9 bpw bank (layers 18-22 at K = 2):
    // 196,041,768,960; either way the sidecar is the layer table's span and a record's spans are its segments'
    const last = bank.layers[bank.layers.len - 1];
    try testing.expectEqual(last.base_offset + 384 * last.record_bytes, bank.sidecar_size);
    for (bank.layers) |l| try testing.expect(l.k == 3 or l.k == 2);
    const l13 = bank.layers[13];
    try testing.expectEqual(@as(u32, 3), l13.k);
    const s = bank.spans(13, 17);
    try testing.expectEqual(l13.base_offset + 17 * l13.record_bytes, s.gu_offset);
    try testing.expectEqual(s.gu_offset + 8_877_056, s.down_offset);
    if (bank.layers[18].k == 2) {
        const l18 = bank.layers[18];
        const s2 = bank.spans(18, 5);
        try testing.expectEqual(@as(u64, 8_892_416), l18.record_bytes);
        try testing.expectEqual(l18.base_offset + 5 * l18.record_bytes, s2.gu_offset);
        try testing.expectEqual(s2.gu_offset + 5_927_936, s2.down_offset);
    }
    const fixture = std.mem.span(env.get(.DSV41_PHASE0_FIXTURE) orelse return);
    const Rec = struct { layer: u32, expert: u32, gu: [2]u64, down: [2]u64 };
    const Fixture = struct { layer_set: []const Rec, pick_set: []const Rec };
    const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, fixture, testing.allocator, .limited(4 << 20));
    defer testing.allocator.free(text);
    const parsed = try std.json.parseFromSlice(Fixture, testing.allocator, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for ([_][]const Rec{ parsed.value.layer_set, parsed.value.pick_set }) |set| for (set) |r| {
        const got = bank.spans(r.layer, r.expert);
        try testing.expectEqual(r.gu[0], got.gu_offset);
        try testing.expectEqual(r.down[0], got.down_offset);
        try testing.expectEqual(r.down[0] - r.gu[0], r.gu[1]);
    };
}
