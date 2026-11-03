//! GLM-5.3's affine expert bank: the pack's `experts.bin` and its `expert-manifest-affine-v1.json` (the contract the
//! `scripts/glm53_pack.py affine` writes) as a bank module for the expert stream (`sdk_ext.expert.assertBank`).
//! One record = one routed expert of one MoE layer: nine segments in the source tensors' bytes, gate / up / down each
//! `weight` U32 `[out, in * bits / 32]`, `scales` and `biases` BF16 `[out, in / 64]`; the gate/up span is the first six.
//! Every rule runs once in `Bank.open`, which also opens the sidecar the read pool reads through; a mismatch refuses
//! the whole bank by name. The slot arrays bind to MLX's own `gather_qmm` (`sdk_ext.quant.GatherQmm`).

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const expert_io = sdk_ext.expert.io;
const bank_file = sdk_ext.expert.bank_file;
const quant = sdk_ext.quant;
const glm = @import("glm_moe_dsa.zig");

pub const format = "mlx-stream-expert-manifest-affine-v1";
pub const manifest_file = "expert-manifest-affine-v1.json";
pub const record_alignment: u64 = 4096;

// ── the bank contract (`sdk_ext.expert.assertBank`) ──

pub const Component = enum(u8) {
    gate_weight,
    gate_scales,
    gate_biases,
    up_weight,
    up_scales,
    up_biases,
    down_weight,
    down_scales,
    down_biases,

    /// The manifest's component name.
    pub fn name(c: Component) []const u8 {
        return names[@intFromEnum(c)];
    }

    const names = [_][]const u8{ "gate.weight", "gate.scales", "gate.biases", "up.weight", "up.scales", "up.biases", "down.weight", "down.scales", "down.biases" };
};
pub const n_components = 9;
pub const gu_components = 6;
pub const Records = expert_io.Records(n_components, gu_components);
pub const routed_top_k = glm.routed_top_k;

pub const Dtype = enum { U32, BF16 };

pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .U32 => .uint32,
        .BF16 => .bfloat16,
    };
}

pub const Segment = struct {
    /// Relative to the record start.
    offset: u64,
    length: u64,
    dtype: Dtype,
    shape: [3]u64 = @splat(0),
    rank: u8,
};

pub const Layer = struct {
    /// The model layer this routed layer is (`mlp_layer_types[layer] == "sparse"`).
    model_layer: u32,
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    segments: [n_components]Segment,
};

/// One projection's slot arrays as MLX's `gather_qmm` binds them: w u32 [rows, out, in * bits / 32], scales and
/// biases bf16 [rows, out, in / 64].
pub const Arrays = quant.GatherQmm.Arrays(mlx.mlx_array);
pub const BankArrays = quant.BankArrays(Arrays);

pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .w = x[0], .scales = x[1], .biases = x[2] },
        .up = .{ .w = x[3], .scales = x[4], .biases = x[5] },
        .down = .{ .w = x[6], .scales = x[7], .biases = x[8] },
    };
}

/// A record's two read ranges.
pub const Spans = struct { gu_offset: u64, down_offset: u64 };

/// A bank's slot geometry as the bill and the module read it (`Bank.geometryOf`): one stream layer per routed layer,
/// one record per expert, one row per fill unit.
pub const Geometry = glm.Geometry;
pub const LayerSlots = glm.LayerSlots;

/// The segment table of one record at `bits`, hidden `hidden`, intermediate `inter` (the contract's order), and its
/// logical bytes; null for dims that do not pack at group 64.
pub fn layerSegments(bits: u32, hidden: u64, inter: u64) ?[n_components]Segment {
    if (bits * hidden % 32 != 0 or bits * inter % 32 != 0 or hidden % glm.group_size != 0 or inter % glm.group_size != 0) return null;
    var segs: [n_components]Segment = undefined;
    var off: u64 = 0;
    for (0..3) |p| {
        const out: u64 = if (p == 2) hidden else inter;
        const in: u64 = if (p == 2) inter else hidden;
        const parts = [3]Segment{
            .{ .offset = 0, .length = out * (in * bits / 32) * 4, .dtype = .U32, .shape = .{ out, in * bits / 32, 0 }, .rank = 2 },
            .{ .offset = 0, .length = out * (in / glm.group_size) * 2, .dtype = .BF16, .shape = .{ out, in / glm.group_size, 0 }, .rank = 2 },
            .{ .offset = 0, .length = out * (in / glm.group_size) * 2, .dtype = .BF16, .shape = .{ out, in / glm.group_size, 0 }, .rank = 2 },
        };
        for (parts, 0..) |sg, j| {
            segs[p * 3 + j] = sg;
            segs[p * 3 + j].offset = off;
            off += sg.length;
        }
    }
    return segs;
}

pub fn logicalBytes(segs: *const [n_components]Segment) u64 {
    return segs[n_components - 1].offset + segs[n_components - 1].length;
}

/// A record's bytes in the sidecar: its logical bytes rounded up to the alignment.
pub fn recordBytes(logical: u64) u64 {
    return std.mem.alignForward(u64, logical, record_alignment);
}

pub const Diag = @import("sdk").Diag;

pub const Refusal = error{
    BankDirNotAbsolute,
    ManifestMissing,
    ManifestSyntax,
    ManifestFormat,
    ModelType,
    QuantNotImplemented,
    DimsMismatch,
    LayerGeometry,
    SidecarGeometry,
    SidecarMissing,
    SidecarOpen,
    RecordCount,
    RecordGeometry,
    RecordDuplicate,
    RecordSha256,
    ParityNotPassed,
};

const refuse = bank_file.refuse;

const SegJson = struct { component: []const u8, dtype: []const u8, shape: []const u64, offset: u64, length: u64 };
const LayerJson = struct { layer: u64, index: u64, record_bytes: u64, logical_bytes: u64, base_offset: u64, segments: []const SegJson };
const RecordJson = struct { layer: u64, index: ?u64 = null, expert: u64, sidecar_offset: u64, record_bytes: u64, logical_bytes: u64, sha256: []const u8 };
const ManifestJson = struct {
    format: []const u8,
    model_type: ?[]const u8 = null,
    quantization: struct { mode: []const u8, bits: u64, group_size: u64 },
    dims: struct { hidden: u64, inter: u64, n_experts: u64, n_layers: u64 },
    components: ?[]const []const u8 = null,
    layers: []const LayerJson,
    sidecar: struct { file: []const u8, alignment: u64, size: u64 },
    records: []const RecordJson,
    parity: struct { all_pass: bool },
};

pub const Bank = struct {
    allocator: std.mem.Allocator,
    bits: u32 = 0,
    hidden: u64 = 0,
    inter: u64 = 0,
    n_experts: u32 = 0,
    /// The routed layers in model order: the bank's layer index is the stream's layer.
    layers: []Layer = &.{},
    /// Each layer's slots (`geometryOf`): its logical bytes a row, its experts, one row a unit.
    slots: []LayerSlots = &.{},
    /// sha256 of each record's logical bytes, [layer * n_experts + expert]; stored, not verified at open
    /// (`scripts/glm53_pack.py --verify` does).
    digests: [][32]u8 = &.{},
    sidecar_path: [:0]const u8 = "",
    /// experts.bin, opened once for the reader (`openUncached`: read-only, O_NOFOLLOW, F_NOCACHE, no read-ahead).
    sidecar: expert_io.UncachedFd = .{ .fd = -1, .size = 0 },

    /// Parses `dir`'s manifest, checks it against the model's config (rules 1-6 of the pack's contract) and opens
    /// experts.bin, once.
    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Bank {
        return bank_file.open(Bank, ManifestJson, manifest_file, allocator, io, dir, c, diag, true);
    }

    /// The bank's geometry from its manifest alone (every rule but the sidecar's file), for the bill: no reader,
    /// no descriptor. Its layers in `allocator`.
    pub fn geometry(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Geometry {
        var b = try bank_file.open(Bank, ManifestJson, manifest_file, allocator, io, dir, c, diag, false);
        defer b.deinit();
        var g = b.geometryOf();
        g.layers = try allocator.dupe(LayerSlots, g.layers);
        return g;
    }

    /// What the bill and the stream's options read of a bank: each routed layer's slots, and its widest record's bytes
    /// and read span (`Stream.stagingBytes`). The layers borrow the bank's.
    pub fn geometryOf(self: *const Bank) Geometry {
        var g: Geometry = .{ .layers = self.slots, .widest_record = 0, .widest_span = 0 };
        for (self.layers) |l| {
            g.widest_record = @max(g.widest_record, l.logical_bytes);
            const gu = l.segments[gu_components].offset;
            g.widest_span = @max(g.widest_span, @max(gu, l.logical_bytes - gu));
        }
        // One record per expert: a routed id's span is its record; one scratch, at the widest layer's arrays.
        g.spec_record = g.widest_record;
        for (self.slots) |sl| {
            if (sl.record_bytes == g.widest_record) {
                g.scratch[0] = sl.array_bytes;
                break;
            }
        }
        return g;
    }

    pub fn deinit(self: *Bank) void {
        if (self.sidecar.fd >= 0) self.sidecar.close();
        self.allocator.free(self.layers);
        self.allocator.free(self.slots);
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
        return .{ .gu_offset = off, .down_offset = off + self.layers[layer].segments[gu_components].offset };
    }

    pub fn digest(self: *const Bank, layer: u32, expert: u32) *const [32]u8 {
        return &self.digests[@as(usize, layer) * self.n_experts + expert];
    }

    /// The bank's description for the gather quant's claim (`sdk_ext.quant.BankPeek`): its `quantization` object
    /// and dims, in `a`.
    pub fn peek(self: *const Bank, a: std.mem.Allocator) !quant.BankPeek {
        const q = try std.fmt.allocPrint(a, "{{\"mode\":\"affine\",\"bits\":{d},\"group_size\":{d}}}", .{ self.bits, glm.group_size });
        return bank_file.peek(a, q, self.hidden, self.inter, self.n_experts, self.layers.len);
    }

    /// The pack's rules against the model's config (`bank_file.open`); returns the records' bytes.
    pub fn adopt(self: *Bank, _: std.mem.Allocator, m: *const ManifestJson, c: *const glm.Config, diag: ?*Diag) (Refusal || std.mem.Allocator.Error)!u64 {
        const a = self.allocator;
        // 1. The format.
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, manifest_file ++ ": format \"{s}\" is not " ++ format, .{m.format});
        if (m.model_type) |t| if (!std.mem.eql(u8, t, glm.model_type)) return refuse(diag, error.ModelType, manifest_file ++ ": model_type \"{s}\" is not " ++ glm.model_type, .{t});
        // 2. The quantization and the dims against config.json.
        const q = m.quantization;
        if (!std.mem.eql(u8, q.mode, "affine") or (q.bits != 3 and q.bits != 4) or q.group_size != glm.group_size)
            return refuse(diag, error.QuantNotImplemented, manifest_file ++ ": quantization {s} at {d} bits, group {d} (this build reads affine at 3 or 4 bits, group 64)", .{ q.mode, q.bits, q.group_size });
        const d = m.dims;
        const n_layers = c.nSparse();
        if (d.hidden != c.hidden_size or d.inter != c.moe_intermediate_size or d.n_experts != c.n_routed_experts or d.n_layers != n_layers)
            return refuse(diag, error.DimsMismatch, manifest_file ++ ": dims hidden {d} / inter {d} / {d} experts / {d} layers, config.json {d} / {d} / {d} / {d}", .{ d.hidden, d.inter, d.n_experts, d.n_layers, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts, n_layers });
        if (m.components) |cs| {
            if (cs.len != n_components) return refuse(diag, error.LayerGeometry, manifest_file ++ ": {d} components, the record has {d}", .{ cs.len, n_components });
            for (cs, 0..) |s, i| if (!std.mem.eql(u8, s, Component.names[i])) return refuse(diag, error.LayerGeometry, manifest_file ++ ": component {d} is \"{s}\", want \"{s}\"", .{ i, s, Component.names[i] });
        }
        var sparse0: [glm.max_layers]u32 = undefined;
        var pb: [96]u8 = undefined;
        const expert_path = std.fmt.bufPrint(&pb, "model.layers.{d}.mlp.switch_mlp.gate_proj", .{c.sparseLayers(&sparse0)[0]}) catch unreachable;
        if (q.bits != c.quant.bitsOf(expert_path)) return refuse(diag, error.QuantNotImplemented, manifest_file ++ ": {d}-bit records, config.json quantizes the experts at {d} bits", .{ q.bits, c.quant.bitsOf(expert_path) });
        self.bits = @intCast(q.bits);
        self.hidden = d.hidden;
        self.inter = d.inter;
        self.n_experts = @intCast(d.n_experts);
        // 3. The layers: the sparse layers in order, each segment as the contract derives it, offsets contiguous.
        const want = layerSegments(self.bits, d.hidden, d.inter) orelse return refuse(diag, error.DimsMismatch, manifest_file ++ ": hidden {d} / inter {d} do not pack at {d} bits, group 64", .{ d.hidden, d.inter, self.bits });
        const logical = logicalBytes(&want);
        const record = recordBytes(logical);
        if (m.layers.len != n_layers) return refuse(diag, error.LayerGeometry, manifest_file ++ ": {d} layer entries, config.json has {d} sparse layers", .{ m.layers.len, n_layers });
        var sparse: [glm.max_layers]u32 = undefined;
        const model_layers = c.sparseLayers(&sparse);
        self.layers = try a.alloc(Layer, n_layers);
        for (m.layers, self.layers, model_layers, 0..) |lj, *l, ml, i| {
            if (lj.layer != ml or lj.index != i) return refuse(diag, error.LayerGeometry, manifest_file ++ ": layer entry {d} is layer {d} index {d}, want sparse layer {d} at index {d}", .{ i, lj.layer, lj.index, ml, i });
            if (lj.segments.len != n_components) return refuse(diag, error.LayerGeometry, manifest_file ++ ": layer {d} has {d} segments", .{ ml, lj.segments.len });
            for (lj.segments, want, 0..) |sj, w, k| {
                const dt = std.meta.stringToEnum(Dtype, sj.dtype);
                if (!std.mem.eql(u8, sj.component, Component.names[k]) or dt == null or dt.? != w.dtype or sj.offset != w.offset or sj.length != w.length or
                    !std.mem.eql(u64, sj.shape, w.shape[0..w.rank]))
                    return refuse(diag, error.LayerGeometry, manifest_file ++ ": layer {d} segment {d} ({s} {s} {any} at {d}, {d} B) differs from the contract's {s} {t} {any} at {d}, {d} B", .{ ml, k, sj.component, sj.dtype, sj.shape, sj.offset, sj.length, Component.names[k], w.dtype, w.shape[0..w.rank], w.offset, w.length });
            }
            const base = @as(u64, i) * d.n_experts * record;
            if (lj.logical_bytes != logical or lj.record_bytes != record or lj.base_offset != base)
                return refuse(diag, error.LayerGeometry, manifest_file ++ ": layer {d} logical / record / base {d} / {d} / {d}, want {d} / {d} / {d}", .{ ml, lj.logical_bytes, lj.record_bytes, lj.base_offset, logical, record, base });
            l.* = .{ .model_layer = ml, .record_bytes = record, .logical_bytes = logical, .base_offset = base, .segments = want };
        }
        self.slots = try a.alloc(LayerSlots, n_layers);
        for (self.slots, self.layers) |*sl, l| {
            sl.* = .{ .record_bytes = l.logical_bytes, .n_records = self.n_experts };
            for (&sl.array_bytes, l.segments) |*x, sg| x.* = sg.length;
        }
        // 4. The sidecar.
        const sc = m.sidecar;
        const total = @as(u64, n_layers) * d.n_experts * record;
        if (!bank_file.plainName(sc.file)) return refuse(diag, error.SidecarGeometry, manifest_file ++ ": sidecar file \"{s}\" is not a plain file name", .{sc.file});
        if (sc.alignment != record_alignment) return refuse(diag, error.SidecarGeometry, manifest_file ++ ": sidecar alignment {d}, want {d}", .{ sc.alignment, record_alignment });
        if (sc.size != total) return refuse(diag, error.SidecarGeometry, manifest_file ++ ": sidecar size {d}, want {d} layers x {d} experts x {d} B = {d}", .{ sc.size, n_layers, d.n_experts, record, total });
        // 5. The records: every (layer, expert) once, at its offset.
        const n_rec = @as(usize, n_layers) * self.n_experts;
        if (m.records.len != n_rec) return refuse(diag, error.RecordCount, manifest_file ++ ": {d} records, want {d} x {d}", .{ m.records.len, n_layers, self.n_experts });
        self.digests = try a.alloc([32]u8, n_rec);
        const seen = try a.alloc(bool, n_rec);
        defer a.free(seen);
        @memset(seen, false);
        for (m.records) |r| {
            const li = for (model_layers, 0..) |ml, i| {
                if (ml == r.layer) break i;
            } else return refuse(diag, error.RecordGeometry, manifest_file ++ ": record of layer {d}, not a sparse layer", .{r.layer});
            if (r.index) |ix| if (ix != li) return refuse(diag, error.RecordGeometry, manifest_file ++ ": record of layer {d} names index {d}, want {d}", .{ r.layer, ix, li });
            if (r.expert >= self.n_experts) return refuse(diag, error.RecordGeometry, manifest_file ++ ": record ({d}, {d}) out of range", .{ r.layer, r.expert });
            const k = li * self.n_experts + r.expert;
            if (seen[k]) return refuse(diag, error.RecordDuplicate, manifest_file ++ ": record ({d}, {d}) duplicated", .{ r.layer, r.expert });
            seen[k] = true;
            const l = &self.layers[li];
            if (r.sidecar_offset != l.base_offset + r.expert * record or r.record_bytes != record or r.logical_bytes != logical)
                return refuse(diag, error.RecordGeometry, manifest_file ++ ": record ({d}, {d}) at {d} ({d} / {d} B), want {d} ({d} / {d} B)", .{ r.layer, r.expert, r.sidecar_offset, r.record_bytes, r.logical_bytes, l.base_offset + r.expert * record, record, logical });
            if (r.sha256.len != 64) return refuse(diag, error.RecordSha256, manifest_file ++ ": record ({d}, {d}) sha256 is not 64 hex digits", .{ r.layer, r.expert });
            _ = std.fmt.hexToBytes(&self.digests[k], r.sha256) catch return refuse(diag, error.RecordSha256, manifest_file ++ ": record ({d}, {d}) sha256 is not hex", .{ r.layer, r.expert });
        }
        // 6. The converter's parity.
        if (!m.parity.all_pass) return refuse(diag, error.ParityNotPassed, manifest_file ++ ": parity.all_pass is not true", .{});
        return total;
    }
};

// ── synthetic packs (tests) ──

const testing = std.testing;

/// Deterministic, position-dependent bytes (`expert_bank.fillPattern`).
pub const fillPattern = @import("../../sdk/weights/streamed/expert/bank.zig").fillPattern;

/// A synthetic bank: `experts.bin` (every record its own pattern) and its manifest, for a config's sparse layers; each
/// field below perturbs one rule. The geometry is written out here independently of `layerSegments`.
pub const Synth = struct {
    bits: u32 = 4,
    format: []const u8 = format,
    mode: []const u8 = "affine",
    group_size: u64 = 64,
    hidden_delta: u64 = 0,
    layer_index_delta: u64 = 0,
    seg_len_delta: u64 = 0,
    base_delta: u64 = 0,
    sidecar_size_delta: u64 = 0,
    sidecar_alignment: u64 = 4096,
    sidecar_file: []const u8 = "experts.bin",
    drop_record: bool = false,
    dup_record: bool = false,
    record_offset_delta: u64 = 0,
    bad_sha: bool = false,
    all_pass: bool = true,
    truncate_file: bool = false,
    no_manifest: bool = false,
};

/// Writes `s`'s bank for `c` into `dir`; returns the experts.bin image (caller frees).
pub fn writeSynth(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, s: Synth) ![]u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    const h: u64 = c.hidden_size;
    const i_: u64 = c.moe_intermediate_size;
    const names = Component.names;
    var lens: [n_components]u64 = undefined;
    var shapes: [n_components][2]u64 = undefined;
    for (0..3) |p| {
        const out: u64 = if (p == 2) h else i_;
        const in: u64 = if (p == 2) i_ else h;
        shapes[p * 3] = .{ out, in * s.bits / 32 };
        lens[p * 3] = out * in * s.bits / 8;
        shapes[p * 3 + 1] = .{ out, in / 64 };
        lens[p * 3 + 1] = out * (in / 64) * 2;
        shapes[p * 3 + 2] = shapes[p * 3 + 1];
        lens[p * 3 + 2] = lens[p * 3 + 1];
    }
    var logical: u64 = 0;
    for (lens) |n| logical += n;
    const rb = (logical + 4095) / 4096 * 4096;
    var sparse: [glm.max_layers]u32 = undefined;
    const layers = c.sparseLayers(&sparse);
    const ne: u64 = c.n_routed_experts;
    const total = layers.len * ne * rb;
    const bin = try a.alloc(u8, total);
    errdefer a.free(bin);
    @memset(bin, 0);
    const shas = try a.alloc([64]u8, layers.len * ne);
    defer a.free(shas);
    for (0..layers.len) |l| for (0..ne) |e| {
        const off = (l * ne + e) * rb;
        fillPattern(bin[off .. off + logical], 1 + l * 1000 + e);
        var dg: [32]u8 = undefined;
        Sha256.hash(bin[off .. off + logical], &dg, .{});
        shas[l * ne + e] = std.fmt.bytesToHex(dg, .lower);
    };
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    try j.print(a, "{{\"format\":\"{s}\",\"model_type\":\"glm_moe_dsa\",\"source\":{{\"repo\":\"synthetic\",\"revision\":null}},\"quantization\":{{\"mode\":\"{s}\",\"bits\":{d},\"group_size\":{d}}},", .{ s.format, s.mode, s.bits, s.group_size });
    try j.print(a, "\"dims\":{{\"hidden\":{d},\"inter\":{d},\"n_experts\":{d},\"n_layers\":{d}}},\"components\":[", .{ h + s.hidden_delta, i_, ne, layers.len });
    for (names, 0..) |n, k| try j.print(a, "{s}\"{s}\"", .{ if (k == 0) "" else ",", n });
    try j.appendSlice(a, "],\"layers\":[");
    for (layers, 0..) |ml, l| {
        const index = l + (if (l == 1) s.layer_index_delta else 0);
        try j.print(a, "{s}{{\"layer\":{d},\"index\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"base_offset\":{d},\"segments\":[", .{ if (l == 0) "" else ",", ml, index, rb, logical, l * ne * rb + (if (l == 1) s.base_delta else 0) });
        var off: u64 = 0;
        for (0..n_components) |k| {
            const len = lens[k] + (if (l == 1 and k == 4) s.seg_len_delta else 0);
            try j.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"offset\":{d},\"length\":{d}}}", .{ if (k == 0) "" else ",", names[k], if (k % 3 == 0) "U32" else "BF16", shapes[k][0], shapes[k][1], off, len });
            off += lens[k];
        }
        try j.appendSlice(a, "]}");
    }
    try j.print(a, "],\"sidecar\":{{\"file\":\"{s}\",\"alignment\":{d},\"size\":{d}}},\"records\":[", .{ s.sidecar_file, s.sidecar_alignment, total + s.sidecar_size_delta });
    var first = true;
    for (layers, 0..) |ml, l| for (0..ne) |e| {
        if (s.drop_record and l == 1 and e == 3) continue;
        const ee = if (s.dup_record and l == 1 and e == 3) 2 else e;
        var off = (l * ne + ee) * rb;
        if (l == 1 and e == 2) off += s.record_offset_delta;
        const sha: []const u8 = if (s.bad_sha and l == 0 and e == 1) "abc" else &shas[l * ne + ee];
        try j.print(a, "{s}{{\"layer\":{d},\"index\":{d},\"expert\":{d},\"sidecar_offset\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"sha256\":\"{s}\"}}", .{ if (first) "" else ",", ml, l, ee, off, rb, logical, sha });
        first = false;
    };
    try j.print(a, "],\"parity\":{{\"all_pass\":{},\"checked\":{d},\"total\":{d},\"method\":\"bytes-equal-source\"}}}}", .{ s.all_pass, layers.len * ne, layers.len * ne });
    if (!s.no_manifest) try dir.writeFile(io, .{ .sub_path = manifest_file, .data = j.items });
    try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = if (s.truncate_file) bin[0 .. total - 4096] else bin });
    return bin;
}

/// The tiny model's config (`glm_moe_dsa.tinyConfigJson`), parsed with no release pinned (tests).
pub fn tinyConfig(a: std.mem.Allocator) !glm.Config {
    const text = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(text);
    return glm.Config.parse(a, text, null, null);
}

pub const tmpRoot = @import("sdk").testing.tmpRoot;

/// One case of a GLM-5.3 bank's refusal table: the synthetic bank's perturbation, its refusal and words of its message.
pub fn RefusalCase(comptime M: type) type {
    return struct { s: M.Synth, err: anyerror, why: []const u8 };
}

/// A GLM-5.3 bank module's refusal table (`M`: its `Synth`, `writeSynth` and `Bank.open` over the tiny model): each
/// case's bank refused with its error and a message holding `why`, and a relative bank dir refused by name.
pub fn expectRefusals(comptime M: type, cases: []const RefusalCase(M)) !void {
    var c = try tinyConfig(testing.allocator);
    defer c.deinit(testing.allocator);
    for (cases) |cs| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const bin = try M.writeSynth(testing.allocator, testing.io, tmp.dir, &c, cs.s);
        defer testing.allocator.free(bin);
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        const r = M.Bank.open(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag);
        if (r) |ok| {
            var b = ok;
            b.deinit();
            std.debug.print("case {any}: opened\n", .{cs.s});
            return error.TestExpectedError;
        } else |e| {
            testing.expectEqual(cs.err, e) catch |x| {
                std.debug.print("case {any}: {s}\n", .{ cs.s, diag.message() });
                return x;
            };
            testing.expect(std.mem.indexOf(u8, diag.message(), cs.why) != null) catch |x| {
                std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
                return x;
            };
        }
    }
    var diag: Diag = .{};
    try testing.expectError(error.BankDirNotAbsolute, M.Bank.open(testing.allocator, testing.io, "packs/glm", &c, &diag));
}

/// A synthetic bank (`writeSynth`) for the tiny model, opened from a temporary directory (tests).
pub const SynthBank = struct {
    tmp: std.testing.TmpDir,
    config: glm.Config,
    image: []u8,
    bank: Bank,

    pub fn open() !SynthBank {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var c = try tinyConfig(testing.allocator);
        errdefer c.deinit(testing.allocator);
        const image = try writeSynth(testing.allocator, testing.io, tmp.dir, &c, .{});
        errdefer testing.allocator.free(image);
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        const bank = Bank.open(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        return .{ .tmp = tmp, .config = c, .image = image, .bank = bank };
    }

    pub fn close(self: *SynthBank) void {
        self.bank.deinit();
        testing.allocator.free(self.image);
        self.config.deinit(testing.allocator);
        self.tmp.cleanup();
    }
};

comptime {
    sdk_ext.expert.assertBank(@This());
    sdk_ext.expert.checkTopology(n_components, gu_components);
}

test "glm bank: GLM-5.3's record: nine segments of 21,233,664 B, the gate/up span 14,155,776 B, 4096-aligned as is" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const segs = layerSegments(4, 6144, 2048).?;
    try testing.expectEqual(@as(u64, 21_233_664), logicalBytes(&segs));
    try testing.expectEqual(@as(u64, 21_233_664), recordBytes(logicalBytes(&segs)));
    try testing.expectEqual(@as(u64, 14_155_776), segs[gu_components].offset);
    try testing.expectEqualSlices(u64, &.{ 2048, 768 }, segs[0].shape[0..2]);
    try testing.expectEqualSlices(u64, &.{ 2048, 96 }, segs[1].shape[0..2]);
    try testing.expectEqual(@as(u64, 6_291_456), segs[0].length);
    try testing.expectEqual(@as(u64, 393_216), segs[2].length);
    try testing.expectEqualSlices(u64, &.{ 6144, 256 }, segs[6].shape[0..2]);
    try testing.expectEqualSlices(u64, &.{ 6144, 32 }, segs[8].shape[0..2]);
    // 75 layers x 256 experts: the 407.7 GB bank.
    try testing.expectEqual(@as(u64, 407_686_348_800), 75 * 256 * recordBytes(logicalBytes(&segs)));
    // 3 bits (the mixed-3_6 build's experts): 16.5 MB records.
    const s3 = layerSegments(3, 6144, 2048).?;
    try testing.expectEqual(@as(u64, 4_718_592 * 3 + 393_216 * 6), logicalBytes(&s3));
    try testing.expect(layerSegments(4, 6100, 2048) == null);
}

test "glm bank: a clean synthetic bank opens: offsets, spans and digests from its manifest" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var sb = try SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    try testing.expectEqual(@as(usize, 4), b.layers.len);
    try testing.expectEqual(@as(u32, 16), b.n_experts);
    try testing.expectEqual(@as(u32, 1), b.layers[0].model_layer);
    try testing.expectEqual(@as(u32, 4), b.layers[3].model_layer);
    // hidden 128, inter 64, 4 bits: 3 x (4096 + 2 x 256) = 13,824 logical bytes in 16 KiB records.
    try testing.expectEqual(@as(u64, 13_824), b.layers[0].logical_bytes);
    try testing.expectEqual(@as(u64, 16_384), b.layers[0].record_bytes);
    try testing.expectEqual(@as(u64, (16 + 5) * 16_384), b.recordOffset(1, 5));
    try testing.expectEqual(Spans{ .gu_offset = (16 + 5) * 16_384, .down_offset = (16 + 5) * 16_384 + 9216 }, b.spans(1, 5));
    try testing.expectEqual(@as(u64, sb.image.len), b.sidecar.size);
    for (0..4) |l| for (0..16) |e| {
        const off = b.recordOffset(@intCast(l), @intCast(e));
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(sb.image[off..][0..13_824], &d, .{});
        try testing.expectEqualSlices(u8, &d, b.digest(@intCast(l), @intCast(e)));
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try b.peek(arena.allocator());
    try testing.expectEqual(@as(?quant.Priority, .generic), quant.GatherQmm.claims(&p, null));
    var qd: Diag = .{};
    try testing.expectEqual(quant.GatherQmm.Params{ .mode = .affine, .bits = 4, .group_size = 64 }, try quant.GatherQmm.params(&p, .{ .hidden = 128, .inter = 64, .top_k = 8, .n_layers = 4, .act = .swiglu, .input = .bfloat16 }, &qd));
}

test "glm bank: every rule of the pack's contract refuses by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try expectRefusals(@This(), &.{
        .{ .s = .{ .no_manifest = true }, .err = error.ManifestMissing, .why = "not found" },
        .{ .s = .{ .format = "mlx-stream-expert-manifest-affine-v2" }, .err = error.ManifestFormat, .why = "format" },
        .{ .s = .{ .mode = "mxfp4" }, .err = error.QuantNotImplemented, .why = "mxfp4" },
        .{ .s = .{ .bits = 8 }, .err = error.QuantNotImplemented, .why = "8 bits" },
        .{ .s = .{ .group_size = 32 }, .err = error.QuantNotImplemented, .why = "group 32" },
        .{ .s = .{ .hidden_delta = 64 }, .err = error.DimsMismatch, .why = "hidden 192" },
        .{ .s = .{ .layer_index_delta = 1 }, .err = error.LayerGeometry, .why = "layer entry 1" },
        .{ .s = .{ .seg_len_delta = 2 }, .err = error.LayerGeometry, .why = "segment 4" },
        .{ .s = .{ .base_delta = 4096 }, .err = error.LayerGeometry, .why = "logical / record / base" },
        .{ .s = .{ .sidecar_alignment = 512 }, .err = error.SidecarGeometry, .why = "alignment" },
        .{ .s = .{ .sidecar_size_delta = 4096 }, .err = error.SidecarGeometry, .why = "sidecar size" },
        .{ .s = .{ .sidecar_file = "../experts.bin" }, .err = error.SidecarGeometry, .why = "plain file name" },
        .{ .s = .{ .drop_record = true }, .err = error.RecordCount, .why = "records" },
        .{ .s = .{ .dup_record = true }, .err = error.RecordDuplicate, .why = "duplicated" },
        .{ .s = .{ .record_offset_delta = 4096 }, .err = error.RecordGeometry, .why = "record (2, 2)" },
        .{ .s = .{ .bad_sha = true }, .err = error.RecordSha256, .why = "sha256" },
        .{ .s = .{ .all_pass = false }, .err = error.ParityNotPassed, .why = "all_pass" },
        .{ .s = .{ .truncate_file = true }, .err = error.SidecarGeometry, .why = "experts.bin is" },
    });
}

test "glm bank: the bank of a config with other dims is refused at its dims" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var c = try tinyConfig(testing.allocator);
    defer c.deinit(testing.allocator);
    const bin = try writeSynth(testing.allocator, testing.io, tmp.dir, &c, .{});
    defer testing.allocator.free(bin);
    var other = c;
    other.n_routed_experts = 32;
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    try testing.expectError(error.DimsMismatch, Bank.open(testing.allocator, testing.io, try tmpRoot(&tmp, &rbuf), &other, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "16 experts") != null);
}

/// The stream over this bank (`sdk_ext.expert.stream.StreamOf`): the same code the EXL3 bank streams through.
pub const Stream = sdk_ext.expert.stream.StreamOf(@This(), false);

test "glm bank: the stream serves every routed id from a slot holding its record, the pre-read staging sized from the bank" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var sb = try SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    const page = std.heap.pageSize();
    try testing.expectEqual(std.mem.alignForward(u64, 9216, page) + page, Stream.stagingBytes(b));
    const s = try Stream.Stream.init(testing.allocator, b, .{ .rows = &.{ 4, 6, 3, 5 }, .max_route_ids = 16, .transient_rows = 32, .wide_depth = 2, .staging_from_bank = true, .lookahead = .{ .budget = 2 }, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(53);
    var ids: [16]u16 = undefined;
    for (0..60) |step| {
        const l: u32 = @intCast(step % 4);
        const n = 1 + rng.random().uintLessThan(usize, ids.len);
        var k: usize = 0;
        while (k < n) {
            const e = rng.random().uintLessThan(u16, 16);
            if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                ids[k] = e;
                k += 1;
            }
        }
        const r = try s.serve(l, ids[0..n]);
        try s.expectServed(r, ids[0..n], sb.image);
        s.release(r);
    }
    try s.flush();
}

test "glm bank: the preload fills the prompt rows; a later prompt finds the residents a decode left (shrink's keep)" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var sb = try SynthBank.open();
    defer sb.close();
    const b = &sb.bank;
    for ([_]Stream.Residents{ .keep, .forget }) |residents| {
        const s = try Stream.Stream.init(testing.allocator, b, .{ .rows = &.{ 4, 6, 3, 5 }, .max_route_ids = 16, .transient_rows = 32, .wide_depth = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 }, .transient_release = true });
        defer s.deinit();
        var order: [16]u16 = undefined;
        for (&order, 0..) |*e, i| e.* = @intCast(15 - i);
        const fill = struct {
            fn f(o: []const u16, _: u32) []const u16 {
                return o;
            }
        }.f;
        // Each layer's rows take the first experts of the order, read and landed.
        try testing.expectEqual(@as(u64, 4 + 6 + 3 + 5), try s.preload(@as([]const u16, &order), fill));
        for ([_]u32{ 4, 6, 3, 5 }, 0..) |rows, l| for (order[0..rows]) |e| {
            try testing.expect(s.layers[l].policy.slotOf(e) != null);
        };
        // A prompt route of the preloaded experts reads nothing.
        const st0 = s.stats();
        var r = try s.route(1, order[0..6], &.{});
        try testing.expectEqual(@as(u32, 6), r.plan.n_hits);
        s.release(r);
        try s.flush();
        try testing.expectEqual(st0.expert_bytes_read, s.stats().expert_bytes_read);
        // A decode and the return to the prompt phase: the prompt rows' residents stay only under `keep`.
        _ = try s.releaseTransient();
        try s.grow(&.{ 6, 8, 5, 7 });
        _ = try s.shrink(&.{ 4, 6, 3, 5 }, residents);
        _ = try s.regrowTransient();
        r = try s.route(1, order[0..6], &.{});
        try testing.expectEqual(@as(u32, if (residents == .keep) 6 else 0), r.plan.n_hits);
        for (0..r.n_parts) |p| {
            try s.waitGu(r, @intCast(p));
            try s.waitDown(r, @intCast(p));
        }
        s.release(r);
        try s.flush();
    }
}
