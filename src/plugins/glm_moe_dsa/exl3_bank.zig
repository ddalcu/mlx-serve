//! GLM-5.3's EXL3 expert bank: the pack's `experts.bin` and its `expert-manifest-exl3-v1.json` (the contract the
//! `scripts/glm53_pack.py exl3` writes) as a bank module for the
//! expert stream (`sdk_ext.expert.assertBank`).
//!   - A stream layer is one bank layer: a routed layer's experts at one K, its K3 bank layer at `2 r`, its K4 bank
//!     layer at `2 r + 1`. A routed layer of one K (a uniform pack, `k_values` [3]) has an empty bank layer at
//!     `2 r + 1` (`Bank.closeLayer`; its slots `glm.LayerSlots.empty`: no expert, so no row, no route, no read, no
//!     gate, no wave). The MTP layer's bank layers are the draft lane's (`mtp/`), outside the stream.
//!   - A routed id is the model's expert id; a bank layer routes only the experts it holds. Its record is the expert's
//!     `tp` minis (one tensor-parallel rank each: nine segments, code I16 = the trellis, rout F16 = svh, rin F16 =
//!     suh per projection), adjacent in the file and read into adjacent rows of the slot arrays (`Layer.minis`).
//!   - The K3 and K4 bank layers' code rows differ: the stream keeps a transient scratch per geometry.
//! Every rule runs once in `Bank.open`, which also opens the sidecar the read pool reads through; a mismatch refuses
//! the whole bank by name. The slot arrays bind to sushi's EXL3 MoE (`glm_moe_dsa_exl3_quant`).

const std = @import("std");
const mlx = @import("sdk").mlx;
const sdk_ext = @import("../../sdk/weights/streamed/sdk_ext.zig");
const expert_io = sdk_ext.expert.io;
const bank_file = sdk_ext.expert.bank_file;
const quant = sdk_ext.quant;
const glm = @import("glm_moe_dsa.zig");
const affine = @import("bank.zig");
const bill_units = @import("bill.zig");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const sushi = if (host_caps.exl3) @import("mlx_host").exl3.sushi else @compileError("glm_moe_dsa: this host exports no exl3 (host_caps): the reference must sit behind host_caps.glm");

pub const format = "mlx-stream-expert-manifest-exl3-v1";
pub const manifest_file = "expert-manifest-exl3-v1.json";
pub const record_alignment: u64 = 4096;
/// The pack's `mtp/`: the MTP layer's bank layers in the same format (the draft lane's, outside the stream).
pub const mtp_dir = "mtp";
pub const mtp_manifest_file = "mtp-manifest-exl3-v1.json";
pub const mtp_sidecar_file = "mtp-experts.bin";

// ── the bank contract (`sdk_ext.expert.assertBank`) ──

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

    pub const names = [_][]const u8{ "gate_proj.code", "gate_proj.rout", "gate_proj.rin", "up_proj.code", "up_proj.rout", "up_proj.rin", "down_proj.code", "down_proj.rout", "down_proj.rin" };
};
pub const n_components = 9;
pub const gu_components = 6;
pub const Records = expert_io.Records(n_components, gu_components);
pub const routed_top_k = glm.routed_top_k;
/// The K3 and the K4 bank layers keep their own transient scratch (`sdk_ext.expert.stream`).
pub const per_geometry_transients = true;
/// Stream layers per routed layer: its K3 bank layer, then its K4 one.
pub const banks_per_layer: u32 = 2;

/// The trellis is I16 in the source and read as uint16 (sushi's dtype for the same bytes); rout and rin are F16.
pub const Dtype = enum { I16, F16 };

pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
    return switch (d) {
        .I16 => .uint16,
        .F16 => .float16,
    };
}

pub const Segment = struct {
    /// Relative to the mini's record start.
    offset: u64,
    length: u64,
    dtype: Dtype,
    shape: [3]u64 = @splat(0),
    rank: u8,
};

pub const Layer = struct {
    /// The model layer whose experts at `k` this bank layer holds.
    model_layer: u32,
    k: u32,
    /// One mini's record: its stride in the file and its logical bytes (the nine segments).
    record_bytes: u64,
    logical_bytes: u64,
    base_offset: u64,
    /// A routed id's records: the expert's tensor-parallel minis.
    minis: u32,
    /// The experts this bank layer holds: the most slot rows it takes.
    n_held: u32,
    segments: [n_components]Segment,
};

/// One projection's slot arrays as sushi's EXL3 MoE binds them (`sushi.Proj`): trellis uint16 `[minis, in/16,
/// out/16, 16K]`, suh float16 `[minis, in]`, svh float16 `[minis, out]`.
pub fn Proj(comptime T: type) type {
    return struct { trellis: T, suh: T, svh: T };
}
pub const Arrays = Proj(mlx.mlx_array);
pub const BankArrays = quant.BankArrays(Arrays);

/// The slot arrays in component order: code = trellis, rout = svh (the output side), rin = suh (the input side).
pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{
        .gate = .{ .trellis = x[0], .svh = x[1], .suh = x[2] },
        .up = .{ .trellis = x[3], .svh = x[4], .suh = x[5] },
        .down = .{ .trellis = x[6], .svh = x[7], .suh = x[8] },
    };
}

pub const Spans = affine.Spans;
pub const Geometry = glm.Geometry;
pub const LayerSlots = glm.LayerSlots;
pub const Diag = @import("sdk").Diag;

/// Segment `ci`'s shape (gate / up: in = hidden, out = mini; down: in = mini, out = hidden; code = trellis
/// `[in/16, out/16, 16K]`, rout = svh `[out]`, rin = suh `[in]`) and its rank.
pub fn segmentShape(ci: usize, hidden: u64, mini: u64, k: u64) struct { shape: [3]u64, rank: u8 } {
    const down = ci >= 6;
    const in_d = if (down) mini else hidden;
    const out_d = if (down) hidden else mini;
    return switch (ci % 3) {
        0 => .{ .shape = .{ in_d / 16, out_d / 16, 16 * k }, .rank = 3 },
        1 => .{ .shape = .{ out_d, 0, 0 }, .rank = 1 },
        else => .{ .shape = .{ in_d, 0, 0 }, .rank = 1 },
    };
}

/// One mini's nine segments at K `k`, back to back, and their bytes.
pub fn miniSegments(hidden: u64, mini: u64, k: u64) struct { segs: [n_components]Segment, logical: u64 } {
    var segs: [n_components]Segment = undefined;
    var off: u64 = 0;
    for (&segs, 0..) |*sg, ci| {
        const sh = segmentShape(ci, hidden, mini, k);
        var n: u64 = 2;
        for (sh.shape[0..sh.rank]) |x| n *= x;
        sg.* = .{ .offset = off, .length = n, .dtype = if (ci % 3 == 0) .I16 else .F16, .shape = sh.shape, .rank = sh.rank };
        off += n;
    }
    return .{ .segs = segs, .logical = off };
}

pub const Refusal = error{
    BankDirNotAbsolute,
    ManifestMissing,
    ManifestSyntax,
    ManifestFormat,
    ModelType,
    QuantNotImplemented,
    DimsMismatch,
    LayerGeometry,
    ExpertMap,
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
const LayerJson = struct { bank_layer: u64, layer: u64, k: u64, mtp: bool, n_minis: u64, record_bytes: u64, logical_bytes: u64, base_offset: u64, experts: []const u64, segments: []const SegJson };
const RecordJson = struct { bank_layer: u64, mini: u64, expert: u64, rank: u64, sidecar_offset: u64, sha256: []const u8 };
const ManifestJson = struct {
    format: []const u8,
    model_type: []const u8,
    quantization: struct { mode: []const u8, codebook: []const u8, codebook_multiplier: u64, tp_ranks: u64, k_values: []const u64 },
    dims: struct { hidden: u64, inter: u64, mini_inter: u64, n_experts: u64 },
    components: []const []const u8,
    layers: []const LayerJson,
    experts: std.json.ArrayHashMap([]const [2]u64),
    sidecar: struct { file: []const u8, alignment: u64, size: u64 },
    records: []const RecordJson,
    parity: struct { all_pass: bool },
};

/// No local index: the expert is the layer's other bank layer's.
pub const no_local: u16 = std.math.maxInt(u16);

/// The bank layers a manifest holds: the trunk's (each routed layer's K3 and K4 bank layers, then the MTP layer's two,
/// outside the stream), or the MTP layer's two alone (the pack's `mtp/`), as the bank's stream layers 0 and 1.
pub const Holds = enum { trunk, mtp };

/// What `adopt` checks a manifest against: the model's config and the bank layers the manifest holds.
pub const Against = struct { c: *const glm.Config, holds: Holds = .trunk };

pub const Bank = struct {
    allocator: std.mem.Allocator,
    hidden: u64 = 0,
    inter: u64 = 0,
    mini_inter: u64 = 0,
    tp: u32 = 0,
    /// The routed ids' range: the model's experts.
    n_experts: u32 = 0,
    /// The manifest's `quantization.k_values` (bit k): every bank layer's K is one of them.
    k_mask: u16 = 0,
    /// The stream layers: routed layer r's K3 bank layer at 2 r, its K4 bank layer (or an empty one) at 2 r + 1.
    layers: []Layer = &.{},
    /// [stream layer * n_experts + expert]: the expert's index among its bank layer's experts (`no_local`: not there).
    local: []u16 = &.{},
    /// [routed layer * n_experts + expert]: the expert's bank layer of its routed layer (0: the first, 1: the second).
    side: []u1 = &.{},
    /// Each stream layer's slots (`geometryOf`): an expert's minis a row, its experts, its share of the units.
    slots: []LayerSlots = &.{},
    sidecar_path: [:0]const u8 = "",
    sidecar: expert_io.UncachedFd = .{ .fd = -1, .size = 0 },

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Bank {
        return bank_file.open(Bank, ManifestJson, manifest_file, allocator, io, dir, Against{ .c = c }, diag, true);
    }

    /// The pack's `mtp/` (`mdir`) under the trunk's rules: the MTP layer's two bank layers (`Holds.mtp`), the experts
    /// map of the MTP layer, the records, the sidecar opened (the draft lane reads its records through it).
    pub fn openMtp(allocator: std.mem.Allocator, io: std.Io, mdir: []const u8, c: *const glm.Config, diag: ?*Diag) !Bank {
        return bank_file.open(Bank, ManifestJson, mtp_manifest_file, allocator, io, mdir, Against{ .c = c, .holds = .mtp }, diag, true);
    }

    /// The bank's geometry from its manifest alone (no sidecar descriptor), for the bill. Its layers in `allocator`.
    pub fn geometry(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, diag: ?*Diag) !Geometry {
        var b = try bank_file.open(Bank, ManifestJson, manifest_file, allocator, io, dir, Against{ .c = c }, diag, false);
        defer b.deinit();
        var g = b.geometryOf();
        g.layers = try allocator.dupe(LayerSlots, g.layers);
        return g;
    }

    /// What the bill reads (`glm_moe_dsa.Geometry`): each stream layer's slots (one row an expert's minis, its experts,
    /// its share of a unit: its experts over `maxUnits`); one slot of every geometry as the widest record (each
    /// geometry's transient scratch holds its own rows, the bill charges them all at it); the widest mini range; the K
    /// values (the split's key). The layers borrow the bank's.
    pub fn geometryOf(self: *const Bank) Geometry {
        var g: Geometry = .{ .layers = self.slots, .widest_record = 0, .widest_span = 0, .k_mask = self.k_mask };
        if (self.layers.len == 0) return g;
        // One slot of each geometry (each K) of the first routed layer's stream layers: an empty one has its other's.
        for (self.layers[0..banks_per_layer], 0..) |*l, i| if (i == 0 or l.k != self.layers[0].k) {
            g.widest_record += @as(u64, l.minis) * l.logical_bytes;
        };
        for (self.layers) |*l| {
            const gu = l.segments[gu_components].offset;
            g.widest_span = @max(g.widest_span, @max(gu, l.logical_bytes - gu));
            // A routed id's minis back to back in the file (`StreamOf.spanOf`), at the widest layer.
            g.spec_record = @max(g.spec_record, (@as(u64, l.minis) - 1) * l.record_bytes + l.logical_bytes);
            g.minis_per_id = @max(g.minis_per_id, l.minis);
        }
        // A transient scratch per geometry (the stream's): the widest span's arrays, then the other's.
        var widest: usize = 0;
        for (self.slots, 0..) |sl, i| if (sl.record_bytes > self.slots[widest].record_bytes) {
            widest = i;
        };
        g.scratch[0] = self.slots[widest].array_bytes;
        for (self.slots) |sl| if (!std.mem.eql(u64, &sl.array_bytes, &g.scratch[0])) {
            g.scratch[1] = sl.array_bytes;
            break;
        };
        return g;
    }

    pub fn deinit(self: *Bank) void {
        if (self.sidecar.fd >= 0) self.sidecar.close();
        self.allocator.free(self.layers);
        self.allocator.free(self.local);
        self.allocator.free(self.side);
        self.allocator.free(self.slots);
        if (self.sidecar_path.len > 0) self.allocator.free(self.sidecar_path);
        self.* = undefined;
    }

    /// The stream layer that holds expert `e` of routed layer `routed`.
    pub fn streamLayer(self: *const Bank, routed: u32, e: u16) u32 {
        return banks_per_layer * routed + self.side[@as(usize, routed) * self.n_experts + e];
    }

    /// The expert's first mini in the sidecar (its minis follow at `record_bytes`).
    pub fn recordOffset(self: *const Bank, layer: u32, expert: u32) u64 {
        const l = &self.layers[layer];
        const loc = self.local[@as(usize, layer) * self.n_experts + expert];
        std.debug.assert(loc != no_local);
        return l.base_offset + @as(u64, loc) * l.minis * l.record_bytes;
    }

    /// The first mini's read ranges (the stream adds the stride per mini).
    pub fn spans(self: *const Bank, layer: u32, expert: u32) Spans {
        const off = self.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + self.layers[layer].segments[gu_components].offset };
    }

    /// The bank's description for the EXL3 quant's claim (`sdk_ext.quant.BankPeek`), in `a`.
    pub fn peek(self: *const Bank, a: std.mem.Allocator) !quant.BankPeek {
        const q = try std.fmt.allocPrint(a, "{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"tp_ranks\":{d}}}", .{ sushi.format.MCG_MULT, self.tp });
        return bank_file.peek(a, q, self.hidden, self.inter, self.n_experts, self.layers.len);
    }

    /// Routed layer `r`'s stream layers once its `sd` bank layers are read: with one, the second stream layer is empty
    /// (no expert: its slots `glm.LayerSlots.empty`; the first's K, segments and record, so the stream keeps one
    /// transient geometry; its base at the first's end); every expert of the layer in one of them.
    fn closeLayer(self: *Bank, comptime file: []const u8, r: usize, sd: u32, diag: ?*Diag) Refusal!void {
        const ne: usize = self.n_experts;
        const first = banks_per_layer * r;
        if (sd == 1) {
            var l = self.layers[first];
            l.base_offset += @as(u64, l.n_held) * l.minis * l.record_bytes;
            l.n_held = 0;
            self.layers[first + 1] = l;
        }
        for (0..ne) |e| if (self.local[first * ne + e] == no_local and self.local[(first + 1) * ne + e] == no_local)
            return refuse(diag, error.ExpertMap, file ++ ": layer {d}: expert {d} in neither bank layer", .{ self.layers[first].model_layer, e });
    }

    /// The manifest's rules against the model's config (`bank_file.open`, `a` its arena); returns the sidecar's bytes.
    pub fn adopt(self: *Bank, a: std.mem.Allocator, m: *const ManifestJson, x: Against, diag: ?*Diag) (Refusal || std.mem.Allocator.Error)!u64 {
        return switch (x.holds) {
            inline else => |h| self.adoptAs(h, a, m, x.c, diag),
        };
    }

    fn adoptAs(self: *Bank, comptime holds: Holds, a: std.mem.Allocator, m: *const ManifestJson, c: *const glm.Config, diag: ?*Diag) (Refusal || std.mem.Allocator.Error)!u64 {
        const file = comptime switch (holds) {
            .trunk => manifest_file,
            .mtp => mtp_manifest_file,
        };
        const gpa = self.allocator;
        // 1. The format.
        if (!std.mem.eql(u8, m.format, format)) return refuse(diag, error.ManifestFormat, file ++ ": format \"{s}\" is not " ++ format, .{m.format});
        if (!std.mem.eql(u8, m.model_type, glm.model_type)) return refuse(diag, error.ModelType, file ++ ": model_type \"{s}\" is not " ++ glm.model_type, .{m.model_type});
        // 2. The quantization and the dims against config.json.
        const q = m.quantization;
        if (!std.mem.eql(u8, q.mode, "exl3") or !std.mem.eql(u8, q.codebook, "mcg") or q.codebook_multiplier != sushi.format.MCG_MULT or q.tp_ranks == 0)
            return refuse(diag, error.QuantNotImplemented, file ++ ": quantization {s} / {s} / multiplier {d} / {d} ranks (this build reads exl3 mcg at multiplier {d})", .{ q.mode, q.codebook, q.codebook_multiplier, q.tp_ranks, sushi.format.MCG_MULT });
        const d = m.dims;
        if (d.hidden != c.hidden_size or d.inter != c.moe_intermediate_size or d.n_experts != c.n_routed_experts or d.mini_inter * q.tp_ranks != d.inter or d.n_experts % banks_per_layer != 0)
            return refuse(diag, error.DimsMismatch, file ++ ": dims hidden {d} / inter {d} = {d} x {d} / {d} experts, config.json {d} / {d} / {d}", .{ d.hidden, d.inter, q.tp_ranks, d.mini_inter, d.n_experts, c.hidden_size, c.moe_intermediate_size, c.n_routed_experts });
        if (m.components.len != n_components) return refuse(diag, error.LayerGeometry, file ++ ": {d} components, the record has {d}", .{ m.components.len, n_components });
        for (m.components, Component.names) |x, y| if (!std.mem.eql(u8, x, y)) return refuse(diag, error.LayerGeometry, file ++ ": component \"{s}\" where the format has \"{s}\"", .{ x, y });
        self.hidden = d.hidden;
        self.inter = d.inter;
        self.mini_inter = d.mini_inter;
        self.tp = @intCast(q.tp_ranks);
        self.n_experts = @intCast(d.n_experts);
        const ne: usize = self.n_experts;
        // The K values: ascending, each a K the trellis takes.
        for (q.k_values, 0..) |k, i| {
            if (k == 0 or k > 8 or sushi.format.kFromPackedDim(@intCast(16 * k)) == null or (i > 0 and k <= q.k_values[i - 1]))
                return refuse(diag, error.QuantNotImplemented, file ++ ": k_values {any}: not ascending Ks of the trellis", .{q.k_values});
            self.k_mask |= @as(u16, 1) << @intCast(k);
        }
        if (q.k_values.len == 0) return refuse(diag, error.QuantNotImplemented, file ++ ": k_values is empty", .{});
        // 3. The bank layers: every one's segments as the format derives them, back to back, the bases contiguous;
        // the routed layers' first (one per K the layer holds, K ascending, their experts each layer's once), then the
        // MTP layer's. The MTP directory holds the MTP layer's alone, as its one routed layer. A routed layer of one K
        // (a uniform pack: every expert in its one bank layer) still has two stream layers: its second is empty
        // (`closeLayer`).
        var sparse: [glm.max_layers]u32 = undefined;
        const mtp_layer = [1]u32{c.n_layers};
        const routed: []const u32 = switch (holds) {
            .trunk => c.sparseLayers(&sparse),
            .mtp => &mtp_layer,
        };
        const n_stream = routed.len * banks_per_layer;
        self.layers = try gpa.alloc(Layer, n_stream);
        self.local = try gpa.alloc(u16, n_stream * ne);
        @memset(self.local, no_local);
        self.side = try gpa.alloc(u1, routed.len * ne);
        var base: u64 = 0;
        var n_minis: u64 = 0;
        // The routed layer the next bank layer belongs to, and the bank layers it has so far.
        var rl: usize = 0;
        var taken: u32 = 0;
        for (m.layers, 0..) |lj, i| {
            if (rl < routed.len and taken > 0 and lj.layer != routed[rl]) {
                try self.closeLayer(file, rl, taken, diag);
                rl += 1;
                taken = 0;
            }
            const streamed = rl < routed.len;
            if (holds == .mtp and !streamed) return refuse(diag, error.LayerGeometry, file ++ ": {d} bank layers, the MTP layer has at most {d} (one per K)", .{ m.layers.len, banks_per_layer });
            const want_layer: u64 = if (streamed) routed[rl] else c.n_layers;
            const want_mtp = holds == .mtp or !streamed;
            if (lj.bank_layer != i or lj.layer != want_layer or lj.mtp != want_mtp)
                return refuse(diag, error.LayerGeometry, file ++ ": bank layer entry {d} is bank layer {d} of layer {d} (mtp {}), want layer {d} (mtp {})", .{ i, lj.bank_layer, lj.layer, lj.mtp, want_layer, want_mtp });
            if (lj.k == 0 or lj.k > 8 or (self.k_mask >> @intCast(lj.k)) & 1 == 0) return refuse(diag, error.QuantNotImplemented, file ++ ": bank layer {d} at K {d}, not in k_values {any}", .{ i, lj.k, q.k_values });
            if (lj.n_minis != q.tp_ranks * lj.experts.len or lj.base_offset != base)
                return refuse(diag, error.LayerGeometry, file ++ ": bank layer {d}: {d} minis for {d} experts, base {d} (want {d})", .{ i, lj.n_minis, lj.experts.len, lj.base_offset, base });
            const want = miniSegments(d.hidden, d.mini_inter, lj.k);
            if (lj.segments.len != n_components) return refuse(diag, error.LayerGeometry, file ++ ": bank layer {d} has {d} segments", .{ i, lj.segments.len });
            for (lj.segments, want.segs, 0..) |sj, w, ci| {
                const dt = std.meta.stringToEnum(Dtype, sj.dtype);
                if (!std.mem.eql(u8, sj.component, Component.names[ci]) or dt == null or dt.? != w.dtype or sj.offset != w.offset or sj.length != w.length or !std.mem.eql(u64, sj.shape, w.shape[0..w.rank]))
                    return refuse(diag, error.LayerGeometry, file ++ ": bank layer {d} segment {s}: {s} {any} at {d}, {d} B (want {s} {any} at {d}, {d} B)", .{ i, sj.component, sj.dtype, sj.shape, sj.offset, sj.length, @tagName(w.dtype), w.shape[0..w.rank], w.offset, w.length });
            }
            if (lj.logical_bytes != want.logical or lj.record_bytes != std.mem.alignForward(u64, want.logical, record_alignment))
                return refuse(diag, error.LayerGeometry, file ++ ": bank layer {d}: logical {d} / record {d} B, want {d} / {d}", .{ i, lj.logical_bytes, lj.record_bytes, want.logical, std.mem.alignForward(u64, want.logical, record_alignment) });
            for (lj.experts, 0..) |e, j| if (e >= ne or (j > 0 and e <= lj.experts[j - 1]))
                return refuse(diag, error.ExpertMap, file ++ ": bank layer {d}: experts not ascending ids below {d} at {d}", .{ i, ne, j });
            if (streamed) {
                const sl = banks_per_layer * rl + taken;
                if (taken == 1 and lj.k <= self.layers[sl - 1].k) return refuse(diag, error.LayerGeometry, file ++ ": layer {d}'s bank layers at K {d} then {d} (K ascending)", .{ lj.layer, self.layers[sl - 1].k, lj.k });
                for (lj.experts, 0..) |e, j| {
                    if (taken == 1 and self.local[(sl - 1) * ne + e] != no_local) return refuse(diag, error.ExpertMap, file ++ ": layer {d}: expert {d} in both bank layers", .{ lj.layer, e });
                    self.local[sl * ne + e] = @intCast(j);
                    self.side[rl * ne + e] = @intCast(taken);
                }
                self.layers[sl] = .{ .model_layer = @intCast(lj.layer), .k = @intCast(lj.k), .record_bytes = lj.record_bytes, .logical_bytes = lj.logical_bytes, .base_offset = lj.base_offset, .minis = self.tp, .n_held = @intCast(lj.experts.len), .segments = want.segs };
                taken += 1;
                if (taken == banks_per_layer) {
                    try self.closeLayer(file, rl, taken, diag);
                    rl += 1;
                    taken = 0;
                }
            }
            base += lj.n_minis * lj.record_bytes;
            n_minis += lj.n_minis;
        }
        if (rl < routed.len and taken > 0) {
            try self.closeLayer(file, rl, taken, diag);
            rl += 1;
        }
        if (rl < routed.len) return refuse(diag, error.LayerGeometry, file ++ ": {d} bank layers, the {d} routed layers need one per K each", .{ m.layers.len, routed.len });
        self.slots = try gpa.alloc(LayerSlots, n_stream);
        for (self.slots, self.layers[0..n_stream]) |*sl, l| {
            sl.* = .{ .record_bytes = @as(u64, l.minis) * l.logical_bytes, .n_records = l.n_held, .share = @as(f64, @floatFromInt(l.n_held)) / @as(f64, @floatFromInt(ne / banks_per_layer)) };
            for (&sl.array_bytes, l.segments) |*x, sg| x.* = @as(u64, l.minis) * sg.length;
        }
        // 4. The experts map: each routed layer's experts at their bank layer's K and local index.
        for (routed, 0..) |ml, r| {
            var kb: [16]u8 = undefined;
            const key = std.fmt.bufPrint(&kb, "{d}", .{ml}) catch unreachable;
            const ex = m.experts.map.get(key) orelse return refuse(diag, error.ExpertMap, file ++ ": experts has no layer {s}", .{key});
            if (ex.len != ne) return refuse(diag, error.ExpertMap, file ++ ": experts[{s}] lists {d} experts, want {d}", .{ key, ex.len, ne });
            for (ex, 0..) |kl, e| {
                const sl = banks_per_layer * r + self.side[r * ne + e];
                if (kl[0] != self.layers[sl].k or kl[1] != self.local[sl * ne + e])
                    return refuse(diag, error.ExpertMap, file ++ ": experts[{s}][{d}] = [{d}, {d}], its bank layer has it at [{d}, {d}]", .{ key, e, kl[0], kl[1], self.layers[sl].k, self.local[sl * ne + e] });
            }
        }
        // 5. The sidecar.
        const sc = m.sidecar;
        if (!bank_file.plainName(sc.file)) return refuse(diag, error.SidecarGeometry, file ++ ": sidecar file \"{s}\" is not a plain file name", .{sc.file});
        if (sc.alignment != record_alignment or sc.size != base) return refuse(diag, error.SidecarGeometry, file ++ ": sidecar alignment {d}, size {d} (want {d} and the bank layers' {d})", .{ sc.alignment, sc.size, record_alignment, base });
        // 6. The records: every (bank layer, mini) once, at its offset, its expert and rank the layout's.
        if (m.records.len != n_minis) return refuse(diag, error.RecordCount, file ++ ": {d} records, the bank layers have {d} minis", .{ m.records.len, n_minis });
        const seen = try a.alloc(bool, @intCast(n_minis));
        @memset(seen, false);
        const first = try a.alloc(u64, m.layers.len);
        var at: u64 = 0;
        for (m.layers, first) |lj, *f| {
            f.* = at;
            at += lj.n_minis;
        }
        for (m.records) |rj| {
            if (rj.bank_layer >= m.layers.len or rj.mini >= m.layers[rj.bank_layer].n_minis) return refuse(diag, error.RecordGeometry, file ++ ": record ({d}, {d}) out of range", .{ rj.bank_layer, rj.mini });
            const lj = m.layers[rj.bank_layer];
            const k = first[rj.bank_layer] + rj.mini;
            if (seen[k]) return refuse(diag, error.RecordDuplicate, file ++ ": record ({d}, {d}) duplicated", .{ rj.bank_layer, rj.mini });
            seen[k] = true;
            if (rj.sidecar_offset != lj.base_offset + rj.mini * lj.record_bytes or rj.expert != lj.experts[rj.mini / q.tp_ranks] or rj.rank != rj.mini % q.tp_ranks)
                return refuse(diag, error.RecordGeometry, file ++ ": record ({d}, {d}) is expert {d} rank {d} at {d}, want expert {d} rank {d} at {d}", .{ rj.bank_layer, rj.mini, rj.expert, rj.rank, rj.sidecar_offset, lj.experts[rj.mini / q.tp_ranks], rj.mini % q.tp_ranks, lj.base_offset + rj.mini * lj.record_bytes });
            var dg: [32]u8 = undefined;
            if (rj.sha256.len != 64) return refuse(diag, error.RecordSha256, file ++ ": record ({d}, {d}) sha256 is not 64 hex digits", .{ rj.bank_layer, rj.mini });
            _ = std.fmt.hexToBytes(&dg, rj.sha256) catch return refuse(diag, error.RecordSha256, file ++ ": record ({d}, {d}) sha256 is not hex", .{ rj.bank_layer, rj.mini });
        }
        // 7. The converter's parity.
        if (!m.parity.all_pass) return refuse(diag, error.ParityNotPassed, file ++ ": parity.all_pass is not true", .{});
        return base;
    }
};

/// Whether `dir` holds an EXL3 bank (its manifest): the plugin's choice of bank at load.
pub fn present(dir: []const u8) bool {
    var pb: [std.fs.max_path_bytes + 1]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&pb, "{s}/" ++ manifest_file, .{dir}, 0) catch return false;
    var st: std.c.Stat = undefined;
    return std.c.stat(p.ptr, &st) == 0;
}

/// The stream over this bank (`sdk_ext.expert.stream.StreamOf`).
pub const Stream = sdk_ext.expert.stream.StreamOf(@This(), false);

comptime {
    sdk_ext.expert.assertBank(@This());
    sdk_ext.expert.checkTopology(n_components, gu_components);
}

// ── synthetic packs (tests) ──

const testing = std.testing;

/// A synthetic EXL3 bank for `c`: per routed layer a K3 bank layer of its even experts and a K4 one of its odd
/// experts, then the MTP layer's two, `tp` minis per expert of seeded bytes (with `signs`: the suh / svh segments
/// +/-1 in f16), the records, the experts map and the sidecar. With `mtp_only`, the pack's `mtp/` alone: the MTP
/// layer's two bank layers, its experts map, manifest and sidecar, in `mtp_dir`. With `k3_experts` n, each layer's
/// first n experts at K3 and the rest at K4; n = all of them is a uniform pack (the converter's for `k_values` [3]): one
/// K3 bank layer per layer. With `manifest_only`, the manifest alone (every record's sha256 zero, no sidecar written):
/// the real geometry for `Bank.geometry`. Each other field perturbs one rule.
pub const Synth = struct {
    tp: u32 = 4,
    signs: bool = false,
    mtp_only: bool = false,
    k3_experts: ?u32 = null,
    manifest_only: bool = false,
    /// The manifest's `k_values` (null: the bank layers' Ks).
    k_values: ?[]const u8 = null,
    format: []const u8 = format,
    multiplier: u64 = 0,
    hidden_delta: u64 = 0,
    seg_len_delta: u64 = 0,
    base_delta: u64 = 0,
    k_swap: bool = false,
    expert_both: bool = false,
    map_delta: u64 = 0,
    sidecar_size_delta: u64 = 0,
    /// The sidecar the manifest names (null: the file written).
    sidecar_file: ?[]const u8 = null,
    drop_record: bool = false,
    dup_record: bool = false,
    record_rank_delta: u64 = 0,
    bad_sha: bool = false,
    all_pass: bool = true,
    truncate_file: bool = false,
    no_manifest: bool = false,
};

/// Writes `s`'s bank for `c` into `dir`; returns the experts.bin image (caller frees).
pub fn writeSynth(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const glm.Config, s: Synth) ![]u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    const h: u64 = c.hidden_size;
    const mini: u64 = c.moe_intermediate_size / s.tp;
    const ne: u64 = c.n_routed_experts;
    var sparse: [glm.max_layers]u32 = undefined;
    const routed: []const u32 = if (s.mtp_only) &.{} else c.sparseLayers(&sparse);
    const mtp_layer = [1]u32{c.n_layers};
    // The layers the experts map lists: the routed layers', or the MTP layer's in its directory.
    const mapped: []const u32 = if (s.mtp_only) &mtp_layer else routed;
    // Bank layers per layer: one when every expert is at K3.
    const uniform = if (s.k3_experts) |n| n == ne else false;
    const per: usize = if (uniform) 1 else banks_per_layer;
    const n_bank = (routed.len + 1) * per;
    const bin = if (s.mtp_only) mtp_sidecar_file else "experts.bin";
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    var recs: std.ArrayList(u8) = .empty;
    defer recs.deinit(a);
    var image: std.ArrayList(u8) = .empty;
    errdefer image.deinit(a);
    var rng = std.Random.DefaultPrng.init(53);
    const mult = if (s.multiplier != 0) s.multiplier else sushi.format.MCG_MULT;
    try j.print(a, "{{\"format\":\"{s}\",\"model_type\":\"glm_moe_dsa\",\"source\":{{\"repo\":\"synthetic\",\"revision\":null}},\"quantization\":{{\"mode\":\"exl3\",\"codebook\":\"mcg\",\"codebook_multiplier\":{d},\"mcg_scalar\":0,\"k_values\":{s},\"tp_ranks\":{d}}},", .{ s.format, mult, s.k_values orelse if (uniform) "[3]" else "[3,4]", s.tp });
    try j.print(a, "\"dims\":{{\"hidden\":{d},\"inter\":{d},\"mini_inter\":{d},\"n_experts\":{d},\"n_model_layers\":{d},\"n_bank_layers\":{d}}},\"components\":[", .{ h + s.hidden_delta, c.moe_intermediate_size, mini, ne, routed.len + 1, n_bank });
    for (Component.names, 0..) |n, i| try j.print(a, "{s}\"{s}\"", .{ if (i == 0) "" else ",", n });
    try j.appendSlice(a, "],\"layers\":[");
    var base: u64 = 0;
    var first_rec = true;
    for (0..n_bank) |bl| {
        const r = bl / per;
        const ml: u64 = if (r < routed.len) routed[r] else c.n_layers;
        var sd = bl % per;
        if (s.k_swap and bl == 2) sd = 1;
        if (s.k_swap and bl == 3) sd = 0;
        const k: u64 = if (sd == 0) 3 else 4;
        const held = (bl % per) == 0;
        var members: std.ArrayList(u64) = .empty;
        defer members.deinit(a);
        for (0..ne) |e| {
            const at_k3 = if (s.k3_experts) |n| e < n else e % 2 == 0;
            if (at_k3 == held or (s.expert_both and bl == 1 and e == 0)) try members.append(a, e);
        }
        const ms = miniSegments(h, mini, k);
        const rb = std.mem.alignForward(u64, ms.logical, record_alignment);
        const n_minis = s.tp * members.items.len;
        try j.print(a, "{s}{{\"bank_layer\":{d},\"layer\":{d},\"k\":{d},\"mtp\":{},\"n_minis\":{d},\"record_bytes\":{d},\"logical_bytes\":{d},\"base_offset\":{d},\"experts\":[", .{ if (bl == 0) "" else ",", bl, ml, k, r == routed.len, n_minis, rb, ms.logical, base + (if (bl == 3) s.base_delta else 0) });
        for (members.items, 0..) |e, i| try j.print(a, "{s}{d}", .{ if (i == 0) "" else ",", e });
        try j.appendSlice(a, "],\"segments\":[");
        for (ms.segs, 0..) |sg, ci| {
            try j.print(a, "{s}{{\"component\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[", .{ if (ci == 0) "" else ",", Component.names[ci], @tagName(sg.dtype) });
            for (sg.shape[0..sg.rank], 0..) |x, i| try j.print(a, "{s}{d}", .{ if (i == 0) "" else ",", x });
            try j.print(a, "],\"offset\":{d},\"length\":{d}}}", .{ sg.offset, sg.length + (if (bl == 2 and ci == 4) s.seg_len_delta else 0) });
        }
        try j.appendSlice(a, "]}");
        for (0..n_minis) |mi| {
            var dg: [32]u8 = @splat(0);
            if (!s.manifest_only) {
                const at = image.items.len;
                try image.appendNTimes(a, 0, @intCast(rb));
                const rec = image.items[at..][0..@intCast(ms.logical)];
                rng.random().bytes(rec);
                if (s.signs) for (ms.segs, 0..) |sg, ci| if (ci % 3 != 0) for (0..sg.length / 2) |i| {
                    std.mem.writeInt(u16, rec[sg.offset + 2 * i ..][0..2], if (rng.random().boolean()) 0x3C00 else 0xBC00, .little);
                };
                Sha256.hash(rec, &dg, .{});
            }
            const hex = std.fmt.bytesToHex(dg, .lower);
            if (s.drop_record and bl == 1 and mi == 3) continue;
            const mm = if (s.dup_record and bl == 1 and mi == 3) mi - 1 else mi;
            const rank = mm % s.tp + (if (bl == 2 and mi == 5) s.record_rank_delta else 0);
            try recs.print(a, "{s}{{\"bank_layer\":{d},\"mini\":{d},\"expert\":{d},\"rank\":{d},\"sidecar_offset\":{d},\"sha256\":\"{s}\"}}", .{ if (first_rec) "" else ",", bl, mm, members.items[mm / s.tp], rank, base + mm * rb, if (s.bad_sha and bl == 0 and mi == 1) "abc" else &hex });
            first_rec = false;
        }
        base += n_minis * rb;
    }
    try j.appendSlice(a, "],\"experts\":{");
    for (mapped, 0..) |ml, r| {
        try j.print(a, "{s}\"{d}\":[", .{ if (r == 0) "" else ",", ml });
        for (0..ne) |e| {
            const kl: [2]u64 = if (s.k3_experts) |n| (if (e < n) .{ 3, e } else .{ 4, e - n }) else .{ if (e % 2 == 0) 3 else 4, e / 2 };
            try j.print(a, "{s}[{d},{d}]", .{ if (e == 0) "" else ",", kl[0], kl[1] + (if (r == 1 and e == 5) s.map_delta else 0) });
        }
        try j.appendSlice(a, "]");
    }
    try j.print(a, "}},\"sidecar\":{{\"file\":\"{s}\",\"alignment\":4096,\"size\":{d}}},\"records\":[{s}],\"parity\":{{\"all_pass\":{},\"checked\":0,\"total\":0,\"method\":\"bytes-equal-source\"}}}}", .{ s.sidecar_file orelse bin, base + s.sidecar_size_delta, recs.items, s.all_pass });
    if (s.mtp_only) try dir.createDirPath(io, mtp_dir);
    const out = if (s.mtp_only) try dir.openDir(io, mtp_dir, .{}) else dir;
    defer if (s.mtp_only) out.close(io);
    if (!s.no_manifest) try out.writeFile(io, .{ .sub_path = if (s.mtp_only) mtp_manifest_file else manifest_file, .data = j.items });
    if (!s.manifest_only) try out.writeFile(io, .{ .sub_path = bin, .data = if (s.truncate_file) image.items[0 .. image.items.len - 4096] else image.items });
    return image.toOwnedSlice(a);
}

/// The tiny model's config with `inter` routed intermediate (the kernels' Hadamard blocks need minis of 128).
pub fn tinyConfigInter(a: std.mem.Allocator, inter: u32) !glm.Config {
    const text = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(text);
    var buf: [64]u8 = undefined;
    const t2 = try std.mem.replaceOwned(u8, a, text, "\"moe_intermediate_size\":64,", try std.fmt.bufPrint(&buf, "\"moe_intermediate_size\":{d},", .{inter}));
    defer a.free(t2);
    return glm.Config.parse(a, t2, null, null);
}

pub const tmpRoot = affine.tmpRoot;

test "glm exl3 bank: GLM-5.3's minis: K3 3,578,880 B in 3,579,904 B records, K4 4,758,528 in 4,759,552, the gate/up span first" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const k3 = miniSegments(6144, 512, 3);
    const k4 = miniSegments(6144, 512, 4);
    try testing.expectEqual(@as(u64, 3_578_880), k3.logical);
    try testing.expectEqual(@as(u64, 4_758_528), k4.logical);
    try testing.expectEqual(@as(u64, 3_579_904), std.mem.alignForward(u64, k3.logical, record_alignment));
    try testing.expectEqual(@as(u64, 4_759_552), std.mem.alignForward(u64, k4.logical, record_alignment));
    try testing.expectEqualSlices(u64, &.{ 384, 32, 48 }, k3.segs[0].shape[0..3]);
    try testing.expectEqualSlices(u64, &.{ 32, 384, 64 }, k4.segs[6].shape[0..3]);
    try testing.expectEqual(@as(u64, 2_385_920), k3.segs[gu_components].offset);
    try testing.expectEqualSlices(u64, &.{6144}, k3.segs[7].shape[0..1]);
    try testing.expectEqualSlices(u64, &.{512}, k3.segs[8].shape[0..1]);
    // The box's layer: 148 K3 and 108 K4 experts, 4 minis each: 4,175,429,632 B.
    try testing.expectEqual(@as(u64, 4_175_429_632), 148 * 4 * 3_579_904 + 108 * 4 * 4_759_552);
}

test "glm exl3 bank: a clean synthetic bank opens: the stream layers, each expert's bank layer and its minis' offsets" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    var b = Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer b.deinit();
    // 4 routed layers, 2 bank layers each; hidden 128, minis of 16.
    try testing.expectEqual(@as(usize, 8), b.layers.len);
    try testing.expectEqual(@as(u32, 3), b.layers[2].k);
    try testing.expectEqual(@as(u32, 4), b.layers[3].k);
    try testing.expectEqual(@as(u32, 2), b.layers[3].model_layer);
    try testing.expectEqual(@as(u32, 8), b.layers[3].n_held);
    try testing.expectEqual(@as(u32, 4), b.layers[0].minis);
    try testing.expectEqual(@as(u32, 2), b.streamLayer(1, 6));
    try testing.expectEqual(@as(u32, 3), b.streamLayer(1, 7));
    const ms3 = miniSegments(128, 16, 3);
    const ms4 = miniSegments(128, 16, 4);
    const rb3 = std.mem.alignForward(u64, ms3.logical, 4096);
    const rb4 = std.mem.alignForward(u64, ms4.logical, 4096);
    // Expert 7 of routed layer 1 (stream layer 3, K4): local 3, after layer 0's two bank layers and layer 1's K3.
    const base3 = 8 * 4 * rb3 + 8 * 4 * rb4 + 8 * 4 * rb3;
    try testing.expectEqual(base3 + 3 * 4 * rb4, b.recordOffset(3, 7));
    try testing.expectEqual(Spans{ .gu_offset = base3 + 3 * 4 * rb4, .down_offset = base3 + 3 * 4 * rb4 + ms4.segs[gu_components].offset }, b.spans(3, 7));
    // The sidecar holds the MTP layer's bank layers too.
    try testing.expectEqual(@as(u64, img.len), b.sidecar.size);
    try testing.expectEqual(@as(u64, 5 * 8 * 4 * (rb3 + rb4)), img.len);
    const g = b.geometryOf();
    // Eight stream layers: each an expert's four minis a row, its eight experts, one of them a unit.
    try testing.expectEqual(@as(usize, 8), g.layers.len);
    // Each slot array a row: a segment over the expert's four minis (the bill rounds each array's buffer to its pages).
    var a3: [n_components]u64 = undefined;
    var a4: [n_components]u64 = undefined;
    for (&a3, &a4, ms3.segs, ms4.segs) |*x3, *x4, s3, s4| {
        x3.* = 4 * s3.length;
        x4.* = 4 * s4.length;
    }
    try testing.expectEqual(glm.LayerSlots{ .record_bytes = 4 * ms3.logical, .n_records = 8, .share = 1, .array_bytes = a3 }, g.layers[0]);
    try testing.expectEqual(glm.LayerSlots{ .record_bytes = 4 * ms4.logical, .n_records = 8, .share = 1, .array_bytes = a4 }, g.layers[1]);
    try testing.expectEqual(4 * (ms3.logical + ms4.logical), g.widest_record);
    // A transient scratch per geometry: the K4 layers' arrays (the widest), then the K3 layers'.
    try testing.expectEqual([2][n_components]u64{ a4, a3 }, g.scratch);
    // The bill's fill over the bank's geometry: units of 1/8 of each bank layer's 8 experts; a unit's bound covers both
    // geometries' slots.
    try testing.expectEqual(@as(u32, 8), bill_units.maxUnits(g));
    var rows: [8]u32 = undefined;
    for (bill_units.rowsAt(g, 5, &rows)) |r| try testing.expectEqual(@as(u32, 5), r);
    try testing.expect(present(try tmpRoot(&tmp, &rbuf)));
    // GLM-5.3's split: 148 K3 and 108 K4 experts, units of 1/128 of each, rounded down.
    const s53 = [2]glm.LayerSlots{ .{ .record_bytes = 4 * ms3.logical, .n_records = 148, .share = 148.0 / 128.0 }, .{ .record_bytes = 4 * ms4.logical, .n_records = 108, .share = 108.0 / 128.0 } };
    const g53: glm.Geometry = .{ .layers = &s53, .widest_record = 4 * (ms3.logical + ms4.logical), .widest_span = 0 };
    try testing.expectEqual(@as(u32, 128), bill_units.maxUnits(g53));
    var r2: [2]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 115, 84 }, bill_units.rowsAt(g53, 100, &r2));
    try testing.expectEqualSlices(u32, &.{ 148, 108 }, bill_units.rowsAt(g53, 128, &r2));
    try testing.expectEqualSlices(u32, &.{ 18, 13 }, bill_units.rowsAt(g53, 16, &r2));
}

test "glm exl3 bank: a uniform pack (one K3 bank layer per layer, the converter's for k_values [3]) opens with two stream layers per routed layer, the second empty: no experts, no rows, the first's geometry" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{ .k3_experts = c.n_routed_experts });
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var diag: Diag = .{};
    var b = Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer b.deinit();
    const ms3 = miniSegments(128, 16, 3);
    const rb3 = std.mem.alignForward(u64, ms3.logical, 4096);
    // 4 routed layers of 16 K3 experts, 4 minis each; then the MTP layer's one bank layer.
    try testing.expectEqual(@as(usize, 8), b.layers.len);
    try testing.expectEqual(@as(u64, 5 * 16 * 4 * rb3), img.len);
    try testing.expectEqual(@as(u64, img.len), b.sidecar.size);
    for (0..4) |r| {
        const k3 = b.layers[2 * r];
        const empty = b.layers[2 * r + 1];
        try testing.expectEqual(@as(u32, 3), k3.k);
        try testing.expectEqual(@as(u32, 16), k3.n_held);
        try testing.expectEqual(@as(u32, 0), empty.n_held);
        try testing.expectEqual(k3.k, empty.k);
        try testing.expectEqual(k3.segments, empty.segments);
        try testing.expectEqual(k3.base_offset + 16 * 4 * rb3, empty.base_offset);
        for (0..16) |e| try testing.expectEqual(@as(u32, @intCast(2 * r)), b.streamLayer(@intCast(r), @intCast(e)));
    }
    // Expert 7 of routed layer 1: local 7, after layer 0's 16 experts.
    try testing.expectEqual(16 * 4 * rb3 + 7 * 4 * rb3, b.recordOffset(2, 7));
    const g = b.geometryOf();
    var a3: [n_components]u64 = undefined;
    for (&a3, ms3.segs) |*x, sg| x.* = 4 * sg.length;
    try testing.expectEqual(glm.LayerSlots{ .record_bytes = 4 * ms3.logical, .n_records = 16, .share = 2, .array_bytes = a3 }, g.layers[0]);
    try testing.expectEqual(glm.LayerSlots{ .record_bytes = 4 * ms3.logical, .n_records = 0, .share = 0, .array_bytes = a3 }, g.layers[1]);
    // One geometry: one slot of it the widest record, one transient scratch.
    try testing.expectEqual(4 * ms3.logical, g.widest_record);
    try testing.expectEqual([2][n_components]u64{ a3, @splat(0) }, g.scratch);
    // The bill's fill: units of 1/8 of the layer's 16 experts, two rows a unit; the empty layers take none.
    try testing.expectEqual(@as(u32, 8), bill_units.maxUnits(g));
    var rows: [8]u32 = undefined;
    for (bill_units.rowsAt(g, 5, &rows), 0..) |x, i| try testing.expectEqual(@as(u32, if (i % 2 == 0) 10 else 0), x);
    // GLM-5.3's uniform 3.0 bpw pack: 256 K3 experts a layer, 128 units, 192 rows at 96.
    const s30 = [2]glm.LayerSlots{ .{ .record_bytes = 4 * 3_578_880, .n_records = 256, .share = 2 }, .{ .record_bytes = 4 * 3_578_880, .n_records = 0, .share = 0 } };
    const g30: glm.Geometry = .{ .layers = &s30, .widest_record = 4 * 3_578_880, .widest_span = 0 };
    try testing.expectEqual(@as(u32, 128), bill_units.maxUnits(g30));
    var r2: [2]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 192, 0 }, bill_units.rowsAt(g30, 96, &r2));
    // The stream: one transient geometry; each routed layer's experts served from its K3 stream layer.
    @memset(&rows, 0);
    for (0..4) |r| rows[2 * r] = 2;
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    try testing.expect(s.alt == null);
    var rng = std.Random.DefaultPrng.init(7);
    var ids: [8]u16 = undefined;
    for (0..24) |step| {
        const l: u32 = @intCast(2 * (step % 4));
        const n = 1 + rng.random().uintLessThan(usize, 6);
        var k: usize = 0;
        while (k < n) {
            const e = rng.random().uintLessThan(u16, 16);
            if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                ids[k] = e;
                k += 1;
            }
        }
        const r = try s.serve(l, ids[0..n]);
        try s.expectServed(r, ids[0..n], img);
        s.release(r);
    }
    try s.flush();
}

test "glm exl3 bank: a uniform pack of 256 K3 experts a routed layer: its routes, their gates, the reads ahead and the grow touch no empty layer" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    const split = @import("split.zig");
    const text = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(text);
    const t256 = try std.mem.replaceOwned(u8, a, text, "\"n_routed_experts\":16,", "\"n_routed_experts\":256,");
    defer a.free(t256);
    var c = try glm.Config.parse(a, t256, null, null);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{ .k3_experts = 256 });
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    const g = b.geometryOf();
    try testing.expectEqual(@as(usize, 8), b.layers.len);
    // The prompt's rows at 6 units, decode's at 24: two a unit on the K3 layers, none on the empty ones; the even split
    // (the 3.0 pack's table is GLM-5.3's dims').
    var prompt_rows: [8]u32 = undefined;
    _ = bill_units.rowsAt(g, 6, &prompt_rows);
    var decode_rows: [8]u32 = undefined;
    try testing.expect(split.decodeRows(g, 6, 24, bill_units.grownBytes(g, 6, 24), &decode_rows) == null);
    _ = bill_units.rowsAt(g, 24, &decode_rows);
    for (prompt_rows, decode_rows, 0..) |pr, dr, i| {
        try testing.expectEqual(@as(u32, if (i % 2 == 0) 12 else 0), pr);
        try testing.expectEqual(@as(u32, if (i % 2 == 0) 48 else 0), dr);
    }
    // Event gates on the host word and the lookahead over the next routed layer's 256 experts, as the module runs them.
    const s = try Stream.Stream.init(a, &b, .{ .rows = &prompt_rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .transient_release = true, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 }, .lookahead = .{ .k = 8, .budget = 4, .preread = false }, .event = .{} });
    defer s.deinit();
    // One geometry: one transient scratch.
    try testing.expect(s.alt == null);
    var rng = std.Random.DefaultPrng.init(11);
    var ids: [8]u16 = undefined;
    var scores: [256]f32 = undefined;
    var n_gates: u64 = 0;
    for (0..2) |phase| {
        if (phase == 1) {
            _ = try s.releaseTransient();
            try s.grow(&decode_rows);
            // The grow gives the empty layers no rows of their own.
            for (s.layers, 0..) |*ls, sl| if (sl % 2 == 1) {
                try testing.expectEqual(@as(u32, 0), ls.base.rows);
                try testing.expect(ls.ext == null);
            };
        }
        for (0..32) |step| {
            const l: u32 = @intCast(step % 4);
            const n = 1 + rng.random().uintLessThan(usize, 8);
            var k: usize = 0;
            while (k < n) {
                const e = rng.random().uintLessThan(u16, 256);
                if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                    ids[k] = e;
                    k += 1;
                }
            }
            for (&scores) |*x| x.* = rng.random().float(f32);
            // Every expert's bank layer is its routed layer's K3 one.
            for (ids[0..n]) |e| try testing.expectEqual(2 * l, b.streamLayer(l, e));
            const r = try s.route(2 * l, ids[0..n], &scores);
            if (try s.gate(r)) |gt| n_gates += 1 + gt.n_parts;
            for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
            try s.expectServed(r, ids[0..n], img);
            s.release(r);
            try s.flush();
        }
    }
    try testing.expect(n_gates > 0);
    // Nothing routed on an empty layer (so nothing read or gated for it), every K3 layer routed; no read ahead for an
    // empty layer at any rank, some for the K3 ones.
    for (0..b.layers.len) |sl| {
        const lc = s.layerCounts(@intCast(sl));
        if (sl % 2 == 1) {
            try testing.expectEqual(sdk_ext.expert.LayerCounts{}, lc);
        } else try testing.expect(lc.routes > 0);
    }
    const br = s.lookaheadByRank();
    for (br[1]) |x| try testing.expectEqual([2]u64{ 0, 0 }, x);
    try testing.expect(s.stats().spec_issued > 0);
}

test "glm exl3 bank: at GLM-5.3's geometry the 3.42 bpw pack's and the uniform 3.0 bpw pack's manifests give the bill's slots, its units and the split's packs" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    const split = @import("split.zig");
    const text = try glm.testConfigJson(a, ",\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"lm_head\":{\"group_size\":64,\"bits\":8}");
    defer a.free(text);
    var c = try glm.Config.parse(a, text, &glm.glm53, null);
    defer c.deinit(a);
    // 148 K3 and 108 K4 experts a routed layer (which ones does not change the slots); all 256 at K3.
    inline for (.{ .{ 148, bill_units.box_exl3.geometry, split.Pack.k3_k4 }, .{ 256, bill_units.box_exl3_30.geometry, split.Pack.k3 } }) |x| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        a.free(try writeSynth(a, testing.io, tmp.dir, &c, .{ .k3_experts = x[0], .manifest_only = true }));
        var rbuf: [512]u8 = undefined;
        var diag: Diag = .{};
        const g = Bank.geometry(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, &diag) catch |e| {
            std.debug.print("refused: {s}\n", .{diag.message()});
            return e;
        };
        defer a.free(g.layers);
        const want: glm.Geometry = x[1];
        try testing.expectEqual(@as(usize, 150), g.layers.len);
        for (want.layers, g.layers) |w, l| try testing.expectEqual(w, l);
        try testing.expectEqual(want.widest_record, g.widest_record);
        try testing.expectEqual(want.widest_span, g.widest_span);
        try testing.expectEqual(want.minis_per_id, g.minis_per_id);
        try testing.expectEqual(want.spec_record, g.spec_record);
        try testing.expectEqual(want.scratch, g.scratch);
        try testing.expectEqual(want.k_mask, g.k_mask);
        try testing.expectEqual(x[2], split.packOf(g).?);
        try testing.expectEqual(@as(u32, 128), bill_units.maxUnits(g));
    }
}

test "glm exl3 bank: every rule of the pack's contract refuses by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try affine.expectRefusals(@This(), &.{
        .{ .s = .{ .no_manifest = true }, .err = error.ManifestMissing, .why = "not found" },
        .{ .s = .{ .format = "mlx-stream-expert-manifest-exl3-v2" }, .err = error.ManifestFormat, .why = "format" },
        .{ .s = .{ .multiplier = 0x83DCD12D }, .err = error.QuantNotImplemented, .why = "multiplier" },
        .{ .s = .{ .k_values = "[4,3]" }, .err = error.QuantNotImplemented, .why = "not ascending" },
        .{ .s = .{ .k_values = "[3]" }, .err = error.QuantNotImplemented, .why = "not in k_values" },
        .{ .s = .{ .hidden_delta = 128 }, .err = error.DimsMismatch, .why = "hidden 256" },
        .{ .s = .{ .seg_len_delta = 2 }, .err = error.LayerGeometry, .why = "segment up_proj.rout" },
        .{ .s = .{ .base_delta = 4096 }, .err = error.LayerGeometry, .why = "base" },
        .{ .s = .{ .k_swap = true }, .err = error.LayerGeometry, .why = "K ascending" },
        .{ .s = .{ .expert_both = true }, .err = error.ExpertMap, .why = "both bank layers" },
        .{ .s = .{ .map_delta = 1 }, .err = error.ExpertMap, .why = "experts[2][5]" },
        .{ .s = .{ .sidecar_size_delta = 4096 }, .err = error.SidecarGeometry, .why = "sidecar alignment" },
        .{ .s = .{ .sidecar_file = "../experts.bin" }, .err = error.SidecarGeometry, .why = "plain file name" },
        .{ .s = .{ .drop_record = true }, .err = error.RecordCount, .why = "records" },
        .{ .s = .{ .dup_record = true }, .err = error.RecordDuplicate, .why = "duplicated" },
        .{ .s = .{ .record_rank_delta = 1 }, .err = error.RecordGeometry, .why = "record (2, 5)" },
        .{ .s = .{ .bad_sha = true }, .err = error.RecordSha256, .why = "sha256" },
        .{ .s = .{ .all_pass = false }, .err = error.ParityNotPassed, .why = "all_pass" },
        .{ .s = .{ .truncate_file = true }, .err = error.SidecarGeometry, .why = "experts.bin is" },
    });
}

test "glm exl3 bank: the stream reads each routed expert's minis into adjacent rows of its bank layer's slots, persistent and transient, both geometries" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    // Two persistent experts per bank layer: most loads go to the transient scratch of their geometry.
    var rows: [8]u32 = undefined;
    @memset(&rows, 2);
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    // The K4 bank layers are the widest (the first scratch), the K3 ones the second.
    try testing.expectEqual(@as(u32, 1), s.transient_layer);
    try testing.expectEqual(@as(u32, 0), s.alt.?.layer);
    try testing.expectEqual(@as(u1, 0), s.geom_of[3]);
    try testing.expectEqual(@as(u1, 1), s.geom_of[2]);
    var rng = std.Random.DefaultPrng.init(7);
    var ids: [8]u16 = undefined;
    for (0..40) |step| {
        const l: u32 = @intCast(step % 8);
        const n = 1 + rng.random().uintLessThan(usize, 4);
        var k: usize = 0;
        while (k < n) {
            const e: u16 = @intCast(2 * rng.random().uintLessThan(u16, 8) + l % 2);
            if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                ids[k] = e;
                k += 1;
            }
        }
        const r = try s.serve(l, ids[0..n]);
        try s.expectServed(r, ids[0..n], img);
        s.release(r);
    }
    try s.flush();
}

test "glm exl3 bank: the lookahead reads ahead the next routed layer's experts over both its bank layers, each its own span, claimable until the call's last route" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try affine.tinyConfig(a);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{});
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    var rows: [8]u32 = undefined;
    @memset(&rows, 2);
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = 8, .transient_rows = 16, .wide_depth = 2, .transient_release = true, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 }, .lookahead = .{ .k = 8, .budget = 4, .preread = false } });
    defer s.deinit();
    _ = try s.releaseTransient();
    try s.grow(&rows);
    // Routed layer 0's call: its K3 route held, its K4 route steps with the next layer's scores, in order for experts
    // 5 (K4), 2 (K3), 6 (K3), 7 (K4) of routed layer 1, none resident. The pool's test hold lets two records start:
    // the two speculative threads take 5 and 2 and wait at the start of their first chunk until a claim lets them go;
    // 6 and 7 stay queued when the next call routes.
    var scores: [16]f32 = @splat(0);
    scores[5] = 9;
    scores[2] = 8.5;
    scores[6] = 8;
    scores[7] = 7.5;
    const q3raw = struct {
        extern fn q3ld_test_spec_hold(starts: i32) void;
        extern fn q3ld_test_spec_held(n: i32, timeout_ns: i64) i32;
    };
    q3raw.q3ld_test_spec_hold(2);
    defer q3raw.q3ld_test_spec_hold(-1);
    const r0 = try s.routeHeld(0, &.{ 0, 4 });
    const r1 = try s.route(1, &.{ 1, 3 }, &scores);
    for ([_]*Stream.Route{ r0, r1 }) |r| for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
    s.release(r0);
    s.release(r1);
    try testing.expectEqual(@as(u64, 4), s.stats().spec_issued);
    // The demand reads done, the two speculative threads start (a chunk starts with at most one demand job running):
    // records 5 and 2 wait at the hold, 6 and 7 stay queued.
    try testing.expectEqual(@as(i32, 0), q3raw.q3ld_test_spec_held(2, 10 * std.time.ns_per_s));
    // Routed layer 1's call: its held K3 route claims 2 (in flight) and reads 6 itself (queued); nothing is settled
    // before its K4 route, which claims 5 and reads 7 itself. Every mini's two ranges of 2 and 5 are served from the
    // records (5's read at its K4 span).
    const q0 = try s.routeHeld(2, &.{ 2, 6 });
    const q1 = try s.route(3, &.{ 5, 7 }, &.{});
    for ([_]*Stream.Route{ q0, q1 }) |r| for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
    const st = s.stats();
    try testing.expectEqual(@as(u64, 2), st.claimed);
    try testing.expectEqual(@as(u64, 2), st.spec_cancelled);
    try testing.expectEqual(@as(u64, 0), st.spec_expired);
    try testing.expectEqual(@as(u64, 2 * 4 * 2), st.adopt_ranges);
    for ([_]struct { r: *Stream.Route, l: u32 }{ .{ .r = q0, .l = 2 }, .{ .r = q1, .l = 3 } }) |x| {
        const geom = &b.layers[x.l];
        for (x.r.plan.slotsOf(), [_]u16{ if (x.l == 2) 2 else 5, if (x.l == 2) 6 else 7 }) |slot, e| {
            const off = b.recordOffset(x.l, e);
            for (geom.segments, 0..) |sg, ci| {
                const row = s.slotRow(x.l, slot, @enumFromInt(ci));
                for (0..geom.minis) |mi| try testing.expectEqualSlices(u8, img[off + mi * geom.record_bytes + sg.offset ..][0..sg.length], row[mi * sg.length ..][0..sg.length]);
            }
        }
    }
    // Their top-k positions in the one row (5, 2, 6, 7 at 0 .. 3), each in its own bank layer: each read ahead and routed.
    const br = s.lookaheadByRank();
    for ([_]u16{ 5, 2, 6, 7 }, [_]u1{ 1, 0, 0, 1 }, 0..) |e, sd, r| {
        try testing.expectEqual(@as(u32, 2) + sd, b.streamLayer(1, e));
        try testing.expectEqual([2]u64{ 1, 1 }, br[sd][r]);
        try testing.expectEqual([2]u64{ 0, 0 }, br[1 - sd][r]);
    }
    for (br) |side| for (side[4..]) |x| try testing.expectEqual([2]u64{ 0, 0 }, x);
    s.release(q0);
    s.release(q1);
    try s.flush();
}

/// One gated route of `max_route_ids` ids on stream layer 0, none resident, over a synthetic bank of `tp` minis per id
/// (96 experts, 48 a bank layer): every record read, its gates registered, its rows the file's.
fn gateWidest(tp: u32, records_per_part: u32) !void {
    const a = testing.allocator;
    const text = try glm.tinyConfigJson(a, glm.tiny_quant);
    defer a.free(text);
    var buf: [64]u8 = undefined;
    const t1 = try std.mem.replaceOwned(u8, a, text, "\"moe_intermediate_size\":64,", try std.fmt.bufPrint(&buf, "\"moe_intermediate_size\":{d},", .{128 * tp}));
    defer a.free(t1);
    const t2 = try std.mem.replaceOwned(u8, a, t1, "\"n_routed_experts\":16,", "\"n_routed_experts\":96,");
    defer a.free(t2);
    var c = try glm.Config.parse(a, t2, null, null);
    defer c.deinit(a);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const img = try writeSynth(a, testing.io, tmp.dir, &c, .{ .tp = tp });
    defer a.free(img);
    var rbuf: [512]u8 = undefined;
    var b = try Bank.open(a, testing.io, try tmpRoot(&tmp, &rbuf), &c, null);
    defer b.deinit();
    try testing.expectEqual(tp, b.layers[0].minis);
    const widest = Stream.max_route_ids;
    var ids: [widest]u16 = undefined;
    var n: usize = 0;
    for (0..96) |e| if (n < widest and b.streamLayer(0, @intCast(e)) == 0) {
        ids[n] = @intCast(e);
        n += 1;
    };
    try testing.expectEqual(@as(usize, widest), n);
    var rows: [8]u32 = undefined;
    @memset(&rows, 2);
    const s = try Stream.Stream.init(a, &b, .{ .rows = &rows, .max_route_ids = widest, .transient_rows = widest, .wide_depth = 1, .transient_release = true, .records_per_part = records_per_part, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 2048 }, .lookahead = .{ .k = 8, .budget = 2, .preread = false }, .event = .{ .watchdog_ms = 10_000 } });
    defer s.deinit();
    _ = try s.releaseTransient();
    try s.grow(&rows);
    const r = try s.route(0, &ids, &.{});
    var records: u32 = 0;
    for (r.partsOf()) |p| records += p.n_reads;
    try testing.expectEqual(@as(u32, widest) * tp, records);
    const g = (try s.gate(r)).?;
    try testing.expectEqual(r.n_parts, g.n_parts);
    for (0..r.n_parts) |p| try s.waitDown(r, @intCast(p));
    try s.expectServed(r, &ids, img);
    s.release(r);
    try s.flush();
    try testing.expectEqual(@as(u64, 0), s.stats().gates_forced);
}

test "glm exl3 bank: the gates of the widest route register: 48 ids of 8 minis (768 tickets), and GLM-5.3's 48 ids of 4 minis" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try gateWidest(8, 1);
    try gateWidest(4, 2);
}
