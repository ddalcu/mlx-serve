//! GLM-5.3-Flash's routed experts as an EXL3 K=3 expert bank (the `exl3` expert family): DeepSeek-V4.1's bank-v2
//! format at GLM's dims, read by DeepSeek's reader (`src/expert/bank.zig Bank.open`: `expert-manifest-v2.json`, the v1
//! runtime `expert-manifest.json`, `experts.bin`), the stream's bank module over it (`SDK`), and the checks the pack's
//! quantization must pass before anything is read (`checkPeek`). The pack contract is the converter's
//! (R/mimo/glm5next-exl3-pack-contract.md): bank layer b = model layer - first_moe_layer; per record nine segments,
//! per projection (gate, up, down) its trellis `code` I16 [in / 16, out / 16, 48], `rout` (exl3 svh) F16 [out] and `rin`
//! (exl3 suh) F16 [in], back to back; 9,474,048 B a record at GLM's dims (2,313 pages: no padding). The codebook
//! multiplier lives in the manifest (`quantization.codebook_multiplier`), not in the record: a record carries no
//! global-scale word, so the NVFP4 bank's scale-line isolation has nothing to isolate here (the slot arrays are the
//! segments' rows, each a whole number of 128-byte lines).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const sdk_ext = @import("../sdk_ext.zig");
const quant = sdk_ext.quant;
const expert_bank = @import("../expert/bank.zig");
const xq = @import("../../../quant/exl3/stream/quant.zig");
const kernels = @import("exl3_kernels.zig");
const Module = @This();

/// The pack's `expert_streaming` block (config.json): the family, the bank-v2 format, its two manifests, the quant.
pub const format = expert_bank.format_v2;
pub const manifest_file = "expert-manifest-v2.json";
pub const runtime_manifest_file = "expert-manifest.json";
pub const quant_name = "exl3-k3-mul1";

pub const n_components = expert_bank.n_components;
pub const gu_components = expert_bank.gu_components;
pub const Records = expert_bank.Records;
pub const Component = expert_bank.Component;
pub const Layer = expert_bank.Layer;
pub const Segment = expert_bank.Segment;
pub const routed_top_k = 8;
pub const page = expert_bank.record_alignment;

/// GLM-5.3-Flash's bank the texts decode: mul1 at K = 3 on every layer, hidden 4096, inter 2048, 288 experts, 42 layers.
pub const flash: expert_bank.Implemented = .{ .codebooks = &.{kernels.codebook}, .k = &.{kernels.k_bits}, .hidden = kernels.hidden, .inter = kernels.inter, .n_experts = 288, .n_layers = 42 };
pub const flash_first_moe_layer: u32 = 3;

/// One projection's slot arrays (DeepSeek's `ProjArrays`: code, rout, rin).
pub fn Arrays(comptime T: type) type {
    return xq.ProjArrays(T);
}

pub fn BankArraysOf(comptime T: type) type {
    return quant.BankArrays(Arrays(T));
}

pub const BankArrays = BankArraysOf(mlx.mlx_array);

pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
    return .{ .gate = .{ .code = x[0], .rout = x[1], .rin = x[2] }, .up = .{ .code = x[3], .rout = x[4], .rin = x[5] }, .down = .{ .code = x[6], .rout = x[7], .rin = x[8] } };
}

pub fn mlxDtype(d: expert_bank.Dtype) mlx.mlx_dtype {
    return expert_bank.mlxDtype(d);
}

/// The record geometry, in the field names the source, the bill and the profile read (the NVFP4 bank's `Geometry`).
pub const Geometry = struct {
    hidden_size: u32,
    intermediate_size: u32,
    num_experts: u32,
    first_moe_layer: u32,
    num_moe_layers: u32,
    logical_bytes: u64,
    record_bytes: u64,
    gate_up_bytes: u64,
    total_bytes: u64,
    segments: [n_components]Segment,

    /// The geometry of `layers` layers of `experts` K = 3 records at (hidden, inter): DeepSeek's `layerSegments`.
    pub fn init(hidden: u64, inter: u64, experts: u64, first: u64, layers: u64) !Geometry {
        if (experts == 0 or experts >= std.math.maxInt(u16) or layers == 0 or layers > 4096 or first > 4096) return error.InvalidGeometry;
        const l = expert_bank.layerSegments(kernels.k_bits, hidden, inter) orelse return error.InvalidGeometry;
        const total = std.math.mul(u64, l.record_bytes, experts * layers) catch return error.InvalidGeometry;
        return .{
            .hidden_size = @intCast(hidden),
            .intermediate_size = @intCast(inter),
            .num_experts = @intCast(experts),
            .first_moe_layer = @intCast(first),
            .num_moe_layers = @intCast(layers),
            .logical_bytes = l.logical_bytes,
            .record_bytes = l.record_bytes,
            .gate_up_bytes = l.segments[gu_components].offset,
            .total_bytes = total,
            .segments = l.segments,
        };
    }

    /// One slot row's bytes over the nine slot arrays (no line padding: the bank declares no `slot_line_bytes`).
    pub fn slotRowBytes(g: *const Geometry) u64 {
        var n: u64 = 0;
        for (g.segments) |seg| n += seg.length;
        return n;
    }

    pub fn requireFlash(g: *const Geometry) !void {
        if (g.hidden_size != flash.hidden or g.intermediate_size != flash.inter or g.num_experts != flash.n_experts or g.first_moe_layer != flash_first_moe_layer or g.num_moe_layers != flash.n_layers)
            return error.UnsupportedTopology;
    }
};

/// The bank's quantization as the texts decode it, field by field (the v2 manifest's `quantization`, DeepSeek's quant
/// `claims` at GLM's dims): mode exl3, the mul1 codebook at its multiplier, the tile, the Hadamard, the scales, K = 3 on
/// every layer and the dims `want`. Refused once, by name, before any read.
pub fn checkPeek(v: *const quant.BankPeek, want: expert_bank.Implemented, diag: *sdk.Diag) !void {
    const q = v.quantization;
    const f = xq.format;
    if (!strIs(q, "mode", f.mode)) return refuse(diag, error.Glm5NextExl3Mode, "quantization.mode \"{s}\" (the texts decode {s})", .{ quant.str(q, "mode") orelse "", f.mode });
    if (!strIs(q, "codebook", f.codebook)) return refuse(diag, error.Glm5NextExl3Codebook, "quantization.codebook \"{s}\" (the texts decode {s} only)", .{ quant.str(q, "codebook") orelse "", f.codebook });
    if ((quant.int(q, "codebook_multiplier") orelse -1) != f.multiplier) return refuse(diag, error.Glm5NextExl3Codebook, "quantization.codebook_multiplier {?d} (mul1 is {d})", .{ quant.int(q, "codebook_multiplier"), f.multiplier });
    const tile = quant.obj(q, "tile");
    if ((quant.int(tile, "size") orelse -1) != f.tile_size or !strIs(tile, "layout", f.tile_layout) or !strIs(tile, "bitstream", f.tile_bitstream))
        return refuse(diag, error.Glm5NextExl3Tile, "quantization.tile {?d} / {s} / {s} (the texts read {d} / {s} / {s})", .{ quant.int(tile, "size"), quant.str(tile, "layout") orelse "", quant.str(tile, "bitstream") orelse "", f.tile_size, f.tile_layout, f.tile_bitstream });
    const had = quant.obj(q, "hadamard");
    if ((quant.int(had, "size") orelse -1) != f.hadamard_size or !strIs(had, "order", f.hadamard_order) or !strIs(had, "scale", f.hadamard_scale))
        return refuse(diag, error.Glm5NextExl3Hadamard, "quantization.hadamard {?d} / {s} / {s} (the texts rotate {d} / {s} / {s})", .{ quant.int(had, "size"), quant.str(had, "order") orelse "", quant.str(had, "scale") orelse "", f.hadamard_size, f.hadamard_order, f.hadamard_scale });
    const sc = quant.obj(q, "scales");
    if (!strIs(sc, "rin", f.scale_rin) or !strIs(sc, "rout", f.scale_rout))
        return refuse(diag, error.Glm5NextExl3Scales, "quantization.scales rin {s} / rout {s} (the texts read {s} / {s})", .{ quant.str(sc, "rin") orelse "", quant.str(sc, "rout") orelse "", f.scale_rin, f.scale_rout });
    if (v.hidden != want.hidden or v.inter != want.inter or v.n_experts != want.n_experts or v.n_layers != want.n_layers or v.layers.len != v.n_layers)
        return refuse(diag, error.Glm5NextExl3Dims, "dims hidden {d} / inter {d} / {d} experts / {d} layers ({d} entries) (the texts are instantiated at {d} / {d}, the bank is {d} x {d})", .{ v.hidden, v.inter, v.n_experts, v.n_layers, v.layers.len, want.hidden, want.inter, want.n_experts, want.n_layers });
    for (v.layers, 0..) |l, li| if (l.bits != kernels.k_bits)
        return refuse(diag, error.Glm5NextExl3K, "layer {d} K {d} (the texts decode K = {d} only)", .{ li, l.bits, kernels.k_bits });
}

fn strIs(v: std.json.Value, field: []const u8, want: []const u8) bool {
    return std.mem.eql(u8, quant.str(v, field) orelse return false, want);
}

fn refuse(diag: *sdk.Diag, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    diag.set("glm5_next exl3 bank: " ++ fmt, args);
    return err;
}

/// The bank's geometry for the bill and the admission (pure host): the v2 manifest peeked (records skipped, no sidecar),
/// its quantization checked (`checkPeek`), the geometry its dims give.
pub fn inspect(a: std.mem.Allocator, io: std.Io, dir: []const u8, first_moe_layer: u32, want: expert_bank.Implemented, diag: *sdk.Diag) !Geometry {
    var p = try expert_bank.peek(a, io, dir, diag);
    defer p.deinit();
    try checkPeek(&p.view, want, diag);
    return Geometry.init(p.view.hidden, p.view.inter, p.view.n_experts, first_moe_layer, p.view.n_layers);
}

/// The open bank: DeepSeek's reader over both manifests and `experts.bin` (every record's place and digest, the
/// sidecar opened past the page cache), after `checkPeek`.
pub const Bank = struct {
    owner: expert_bank.Bank,
    geometry: Geometry,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, first_moe_layer: u32, want: expert_bank.Implemented, diag: *sdk.Diag) !Bank {
        const geometry = try inspect(a, io, dir, first_moe_layer, want, diag);
        var owner = try expert_bank.Bank.open(a, io, dir, want, diag);
        errdefer owner.deinit();
        for (owner.layers) |l| for (l.segments, geometry.segments) |x, y| {
            if (x.offset != y.offset or x.length != y.length) return refuse(diag, error.Glm5NextExl3Dims, "a layer's segments differ from the geometry", .{});
        };
        if (owner.sidecar_size != geometry.total_bytes) return refuse(diag, error.Glm5NextExl3Dims, "sidecar {d} B, the geometry's {d} B", .{ owner.sidecar_size, geometry.total_bytes });
        return .{ .owner = owner, .geometry = geometry };
    }

    pub fn deinit(b: *Bank) void {
        b.owner.deinit();
        b.* = undefined;
    }
};

/// The stream's view of an open Bank (the NVFP4 SDKBank's fields): borrowed layers and sidecar.
pub const SDKBank = struct {
    layers: []Layer,
    n_experts: u32,
    first_moe_layer: u32,
    record_bytes: u64,
    sidecar: sdk_ext.expert.UncachedFd,

    pub fn init(_: std.mem.Allocator, owner: *const Bank) !SDKBank {
        if (owner.owner.sidecar.fd < 0) return error.SidecarNotOpen;
        return .{ .layers = owner.owner.layers, .n_experts = owner.owner.n_experts, .first_moe_layer = owner.geometry.first_moe_layer, .record_bytes = owner.geometry.record_bytes, .sidecar = owner.owner.sidecar };
    }

    pub fn deinit(b: *SDKBank) void {
        b.* = undefined;
    }

    /// SDK layer 0 is the bank's layer 0 (model layer `first_moe_layer`).
    pub fn recordOffset(b: *const SDKBank, layer: u32, expert: u32) u64 {
        const l = &b.layers[layer];
        return l.base_offset + @as(u64, expert) * l.record_bytes;
    }

    pub fn spans(b: *const SDKBank, layer: u32, expert: u32) struct { gu_offset: u64, down_offset: u64 } {
        const off = b.recordOffset(layer, expert);
        return .{ .gu_offset = off, .down_offset = off + b.layers[layer].segments[gu_components].offset };
    }
};

/// StreamOf's bank module (no `slot_line_bytes`: nothing to isolate).
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
    /// Every slot array one VM page at least (`StreamOf.slot_array_min_bytes`): the direct reads' 4 KiB block alignment
    /// (a one-row ext of a 4 KiB rin / rout array was refused with DirectReadsRow).
    pub const slot_array_min_bytes: u64 = std.heap.page_size_max;
};

/// The slot arrays the stream binds, checked once per bank at construction: the texts' shapes and dtypes.
pub fn checkArrays(g: anytype, which: []const u8, p: Arrays(@TypeOf(g.*).T), down: bool) !void {
    const d = kernels.projDims(if (down) .down else .gate_up);
    const sc = g.shapeOf(p.code);
    const so = g.shapeOf(p.rout);
    const si = g.shapeOf(p.rin);
    const ok = sc.n == 4 and sc.d[1] == d.code[0] and sc.d[2] == d.code[1] and sc.d[3] == d.code[2] and g.dtypeOf(p.code) == .int16 and
        so.n == 2 and so.d[0] == sc.d[0] and so.d[1] == d.rout and g.dtypeOf(p.rout) == .float16 and
        si.n == 2 and si.d[0] == sc.d[0] and si.d[1] == d.rin and g.dtypeOf(p.rin) == .float16;
    if (!ok) {
        sdk.log.err("glm5_next: {s} slot arrays are {any} {any} {any}, not the exl3 bank's [slots, {d}, {d}, {d}] i16, [slots, {d}] f16, [slots, {d}] f16\n", .{ which, sc.slice(), so.slice(), si.slice(), d.code[0], d.code[1], d.code[2], d.rout, d.rin });
        return error.Glm5NextSlotArrays;
    }
}

// ── Tests ──

const testing = std.testing;

test "glm5_next exl3 bank: Flash's record is nine segments back to back, 9,474,048 B on whole pages, gate/up span first" {
    const g = try Geometry.init(4096, 2048, 288, 3, 42);
    try g.requireFlash();
    try testing.expectEqual(@as(u64, 9_474_048), g.logical_bytes);
    try testing.expectEqual(@as(u64, 9_474_048), g.record_bytes);
    try testing.expectEqual(@as(u64, 2313 * 4096), g.record_bytes);
    try testing.expectEqual(@as(u64, 6_316_032), g.gate_up_bytes);
    try testing.expectEqual(@as(u64, 42 * 288 * 9_474_048), g.total_bytes);
    try testing.expectEqual(@as(u64, 114_598_084_608), g.total_bytes);
    try testing.expectEqual(g.record_bytes, g.slotRowBytes());
    // The contract's table (R/mimo/glm5next-exl3-pack-contract.md sec. 2).
    const want = [_][2]u64{ .{ 0, 3_145_728 }, .{ 3_145_728, 4096 }, .{ 3_149_824, 8192 }, .{ 3_158_016, 3_145_728 }, .{ 6_303_744, 4096 }, .{ 6_307_840, 8192 }, .{ 6_316_032, 3_145_728 }, .{ 9_461_760, 8192 }, .{ 9_469_952, 4096 } };
    for (g.segments, want) |s, w| {
        try testing.expectEqual(w[0], s.offset);
        try testing.expectEqual(w[1], s.length);
        try testing.expectEqual(@as(u64, 0), s.length % 128);
    }
    try testing.expectEqualSlices(u64, &.{ 256, 128, 48 }, g.segments[0].shape[0..3]);
    try testing.expectEqualSlices(u64, &.{ 128, 256, 48 }, g.segments[6].shape[0..3]);
    // The stream's bank contract.
    sdk_ext.expert.assertBank(SDK);
}

/// The synthetic bank's quantization object as the converter writes it (DeepSeek's synth writes only the tile size).
fn fullQuantization(a: std.mem.Allocator, text: []const u8, codebook: []const u8) ![]u8 {
    const from = try std.fmt.allocPrint(a, "\"codebook\":\"{s}\",\"codebook_multiplier\":{d},\"tile\":{{\"size\":16}}", .{ codebook, @as(u64, 0x83DCD12D) });
    defer a.free(from);
    const to = try std.fmt.allocPrint(a, "\"codebook\":\"{s}\",\"codebook_multiplier\":{d},\"tile\":{{\"size\":16,\"layout\":\"exl3-tensor-core\",\"bitstream\":\"msb-first-swap16\"}},\"hadamard\":{{\"size\":128,\"order\":\"sylvester-natural\",\"scale\":\"1/sqrt(128)\"}},\"scales\":{{\"rin\":\"exl3.suh\",\"rout\":\"exl3.svh\"}}", .{ codebook, @as(u64, 0x83DCD12D) });
    defer a.free(to);
    if (std.mem.count(u8, text, from) != 1) return error.TestFixture;
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// A synthetic two-layer, eight-expert bank at GLM's expert shape (DeepSeek's writer at hidden 4096 / inter 2048),
/// its v2 manifest completed (`fullQuantization`).
const Synth = struct {
    tmp: testing.TmpDir,
    root: []const u8,
    rbuf: [512]u8,
    /// The bank's experts.bin as written (`Stream.expectServed`'s image).
    image: []u8,

    fn write(s: expert_bank.Synth, codebook: []const u8) !*Synth {
        const a = testing.allocator;
        const self = try a.create(Synth);
        errdefer a.destroy(self);
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        self.image = try expert_bank.writeSynth(a, &self.tmp, s);
        errdefer a.free(self.image);
        const io = testing.io;
        const v2 = try self.tmp.dir.readFileAlloc(io, manifest_file, a, .limited(64 << 20));
        defer a.free(v2);
        const full = try fullQuantization(a, v2, codebook);
        defer a.free(full);
        try self.tmp.dir.writeFile(io, .{ .sub_path = manifest_file, .data = full });
        self.root = try expert_bank.tmpRoot(&self.tmp, &self.rbuf);
        return self;
    }

    fn close(self: *Synth) void {
        testing.allocator.free(self.image);
        self.tmp.cleanup();
        testing.allocator.destroy(self);
    }
};

const synth_implemented: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 4096, .inter = 2048, .n_experts = 8, .n_layers = 2 };

test "glm5_next exl3 bank: a synthetic two-layer, eight-expert bank at GLM's shape opens: manifests, records, spans, the stream's view" {
    const s = try Synth.write(.{ .hidden = 4096, .inter = 2048, .n_experts = 8 }, "mul1");
    defer s.close();
    var diag: sdk.Diag = .{};
    var bank = Bank.open(testing.allocator, testing.io, s.root, 3, synth_implemented, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bank.deinit();
    const g = bank.geometry;
    try testing.expectEqual(@as(u64, 9_474_048), g.record_bytes);
    try testing.expectEqual(@as(u64, 2 * 8 * 9_474_048), g.total_bytes);
    var sb = try SDKBank.init(testing.allocator, &bank);
    defer sb.deinit();
    try testing.expectEqual(@as(u32, 3), sb.first_moe_layer);
    try testing.expectEqual(@as(u64, (8 + 5) * 9_474_048), sb.recordOffset(1, 5));
    try testing.expectEqual(@as(u64, (8 + 5) * 9_474_048 + 6_316_032), sb.spans(1, 5).down_offset);
    // The record's digest is the manifest's.
    try testing.expect(bank.owner.digest(1, 7).padded[0] != 0 or bank.owner.digest(1, 7).padded[1] != 0);
}

test "glm5_next exl3 bank: K other than 3, a codebook other than mul1, another multiplier or other dims refuse at construction, by name" {
    const a = testing.allocator;
    // Small banks (DeepSeek's synthetic dims 64 / 32, four experts): the checks run on the v2 manifest before any record
    // is read, in this order: codebook, multiplier, the dims, every layer's K.
    const small: expert_bank.Implemented = .{ .codebooks = &.{"mul1"}, .k = &.{3}, .hidden = 64, .inter = 32, .n_experts = 4, .n_layers = 2 };
    const Case = struct { synth: expert_bank.Synth, codebook: []const u8 = "mul1", want: expert_bank.Implemented = small, err: anyerror, says: []const u8 };
    for ([_]Case{
        .{ .synth = .{ .k = &.{ 3, 4 } }, .err = error.Glm5NextExl3K, .says = "layer 1 K 4" },
        .{ .synth = .{ .k = &.{ 2, 3 } }, .err = error.Glm5NextExl3K, .says = "layer 0 K 2" },
        .{ .synth = .{ .codebook = "mcg" }, .codebook = "mcg", .err = error.Glm5NextExl3Codebook, .says = "codebook \"mcg\"" },
        .{ .synth = .{ .multiplier = 0xCBAC1FED }, .err = error.Glm5NextExl3Codebook, .says = "codebook_multiplier" },
        .{ .synth = .{}, .want = flash, .err = error.Glm5NextExl3Dims, .says = "the bank is 288 x 42" },
    }) |c| {
        const s = blk: {
            if (c.synth.multiplier != 0x83DCD12D) {
                // The multiplier case: the writer's text names the other multiplier; complete it by hand.
                var tmp = testing.tmpDir(.{});
                errdefer tmp.cleanup();
                const bin = try expert_bank.writeSynth(a, &tmp, c.synth);
                errdefer a.free(bin);
                const v2 = try tmp.dir.readFileAlloc(testing.io, manifest_file, a, .limited(64 << 20));
                defer a.free(v2);
                const from = "\"tile\":{\"size\":16}";
                const to = "\"tile\":{\"size\":16,\"layout\":\"exl3-tensor-core\",\"bitstream\":\"msb-first-swap16\"},\"hadamard\":{\"size\":128,\"order\":\"sylvester-natural\",\"scale\":\"1/sqrt(128)\"},\"scales\":{\"rin\":\"exl3.suh\",\"rout\":\"exl3.svh\"}";
                const full = try std.mem.replaceOwned(u8, a, v2, from, to);
                defer a.free(full);
                try tmp.dir.writeFile(testing.io, .{ .sub_path = manifest_file, .data = full });
                const self = try a.create(Synth);
                self.tmp = tmp;
                self.image = bin;
                self.root = try expert_bank.tmpRoot(&self.tmp, &self.rbuf);
                break :blk self;
            }
            break :blk try Synth.write(c.synth, c.codebook);
        };
        defer s.close();
        var diag: sdk.Diag = .{};
        try testing.expectError(c.err, Bank.open(a, testing.io, s.root, 3, c.want, &diag));
        try testing.expect(std.mem.startsWith(u8, diag.message(), "glm5_next exl3 bank: "));
        try testing.expect(std.mem.indexOf(u8, diag.message(), c.says) != null);
        // The bill's inspect refuses the same way (before any row is billed).
        var d2: sdk.Diag = .{};
        try testing.expectError(c.err, inspect(a, testing.io, s.root, 3, c.want, &d2));
    }
}

test "glm5_next exl3 bank: direct reads: the pack's record geometry passes the 4 KiB block check (and fails today's 16 KiB page rule); a 1 KiB-misaligned record is refused by name" {
    const S = sdk_ext.expert.stream.StreamOf(SDK, false);
    const io = sdk_ext.expert.io;
    var layers: [42]Layer = undefined;
    const ks: [42]u32 = @splat(3);
    const total = expert_bank.layerTable(&ks, 4096, 2048, 288, &layers).?;
    try testing.expectEqual(@as(u64, 114_598_084_608), total);
    var bank: SDKBank = .{ .layers = &layers, .n_experts = 288, .first_moe_layer = 3, .record_bytes = 9_474_048, .sidecar = .{ .fd = -1, .size = total } };
    // APFS's block (the micro-proof's fstatfs f_bsize on the pack's volume) on the 16 KiB page.
    try io.checkDirectBlock(4096, 16384);
    try S.checkDirectReads(&bank, 4096);
    // Today's 16 KiB page rule refuses the pack: the rout / rin segments are 4 and 8 KiB, and records 1, 2, 3 of a layer
    // sit 4, 8, 12 KiB past a 16 KiB boundary.
    try testing.expectEqual(@as(u64, 4096), bank.recordOffset(5, 41) % 16384);
    try testing.expectError(error.DirectReadsLength, S.checkDirectReads(&bank, 16384));
    // A synthetic bank whose layer 7 starts 1 KiB off the block: refused by name.
    layers[7].base_offset += 1024;
    try testing.expectError(error.DirectReadsOffset, S.checkDirectReads(&bank, 4096));
    layers[7].base_offset -= 1024;
    // A segment of 1 KiB-misaligned length: refused by name.
    layers[9].segments[1].length += 1024;
    try testing.expectError(error.DirectReadsLength, S.checkDirectReads(&bank, 4096));
    layers[9].segments[1].length -= 1024;
    try S.checkDirectReads(&bank, 4096);
    // Blocks the route refuses: not a power of two, under 512 B, over the page.
    for ([_]u64{ 0, 256, 3072, 32768 }) |b| try testing.expectError(error.DirectReadsBlock, io.checkDirectBlock(b, 16384));
    // The rows: host slot rows of Flash's segments (page-aligned banks, 4 / 8 KiB and 3 MiB rows) pass; a 1 KiB-skewed base
    // is refused by name.
    var hr = try S.HostSlotRows.init(&layers[0], 3);
    defer hr.deinit();
    var r: S.Rows = .{ .rows = 3, .row_bytes = hr.row_bytes };
    for (&r.base, hr.banks) |*b, bk| b.* = @intFromPtr(bk.ptr);
    try S.checkDirectRows(&r, 4096);
    r.base[4] += 1024;
    try testing.expectError(error.DirectReadsRow, S.checkDirectRows(&r, 4096));
}

test "glm5_next exl3 bank: direct rows: every slot array of a one-row ext holds a VM page at least, whole 4 KiB blocks (the rin / rout rows are 4 and 8 KiB)" {
    const S = sdk_ext.expert.stream.StreamOf(SDK, false);
    var layers: [42]Layer = undefined;
    const ks: [42]u32 = @splat(3);
    _ = expert_bank.layerTable(&ks, 4096, 2048, 288, &layers).?;
    for (0..n_components) |c| {
        const bytes = S.arrayBytes(&layers[17], c, 1);
        try testing.expect(bytes >= std.heap.page_size_max);
        try testing.expectEqual(@as(u64, 0), bytes % 4096);
        // The MLX shape holds the same bytes (rows past the first are never addressed).
        var shape: [4]c_int = undefined;
        const ndim = try S.arrayShape(&layers[17], c, 1, &shape);
        var n: u64 = 1;
        for (shape[1..ndim]) |d| n *= @intCast(d);
        try testing.expectEqual(bytes, @as(u64, @intCast(shape[0])) * n * 2);
    }
}

test "glm5_next exl3 bank: direct rows: the served phase cycle (prompt, release, grow, shrink, regrow, release, grow) keeps every row on the block; a refused grow fails the stream closed" {
    const Bx = @import("experts.zig").Types(.exl3).Bridge;
    const a = testing.allocator;
    const s = try Synth.write(.{ .hidden = 4096, .inter = 2048, .n_experts = 8 }, "mul1");
    defer s.close();
    var diag: sdk.Diag = .{};
    var bank = try Bank.open(a, testing.io, s.root, 3, synth_implemented, &diag);
    defer bank.deinit();
    var sb = try SDKBank.init(a, &bank);
    defer sb.deinit();
    // The served options (direct demand reads, phase rows, decode's window 0), without the GPU event, the lookahead and
    // the overlap schedule: none of them allocates a row.
    const prompt_rows = [_]u32{ 4, 4 };
    const decode_rows = [_]u32{ 6, 7 };
    var opts = try @import("experts.zig").streamOptionsOf(Bx, &prompt_rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .direct);
    opts.event = null;
    opts.lookahead = null;
    opts.prompt_reads = .stock;
    opts.prompt_ahead = .head;
    opts.prompt_relocate = false;
    opts.ahead_cap = null;
    try testing.expect(opts.pool.direct and opts.transient_release);
    const st = try Bx.Stream.init(a, &sb, opts);
    defer st.deinit();
    try testing.expectEqual(@as(u64, 4096), st.direct_block);
    // A 26-token prompt: 208 routed ids, one wave per layer (8 experts: every id of the wave distinct here).
    const wave = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const dec = [_]u16{ 6, 1, 7, 3 };
    for (0..2) |request| {
        for (0..2) |l| {
            const r = try st.serve(@intCast(l), &wave);
            st.release(r);
        }
        try st.flush();
        _ = try st.releaseTransient();
        try st.grow(&decode_rows);
        try testing.expect(!st.failed);
        for (0..2) |l| {
            const r = try st.serve(@intCast(l), &dec);
            st.release(r);
        }
        try st.flush();
        if (request == 0) {
            // The frees and the regrow return the bytes the arrays hold (`arrayBytes`, the page floor included), not
            // rows x row bytes: the one-to-three-row exts' 4 / 8 KiB components hold a page each.
            const S = sdk_ext.expert.stream.StreamOf(SDK, false);
            var want: u64 = 0;
            var unfloored: u64 = 0;
            for (0..2) |l| for (0..n_components) |c| {
                want += S.arrayBytes(&st.bank.layers[l], c, decode_rows[l] - prompt_rows[l]);
                unfloored += S.rowBytes(&st.bank.layers[l], c) * (decode_rows[l] - prompt_rows[l]);
            };
            for (0..n_components) |c| {
                want += S.arrayBytes(&st.bank.layers[st.transient_layer], c, st.transient.rows);
                unfloored += S.rowBytes(&st.bank.layers[st.transient_layer], c) * st.transient.rows;
            }
            try testing.expect(st.alt == null and want > unfloored);
            try testing.expectEqual(want, try st.shrink(&prompt_rows, .forget));
            try testing.expectEqual(st.promptTransientBytes(), try st.regrowTransient());
            try testing.expect(!st.failed);
        }
    }
    _ = try st.shrink(&prompt_rows, .forget);
    _ = try st.regrowTransient();
    // A grow whose rows land off the block (a prompt row skewed 1 KiB): refused by name, the stream failed closed, every
    // later phase call refused by `StreamFailed`; the rows it installed stay owned (one free at deinit).
    _ = try st.releaseTransient();
    st.layers[1].base.base[2] += 1024;
    try testing.expectError(error.DirectReadsRow, st.grow(&decode_rows));
    st.layers[1].base.base[2] -= 1024;
    try testing.expect(st.failed);
    try testing.expectError(error.StreamFailed, st.shrink(&prompt_rows, .forget));
}

test "glm5_next exl3 bank: prompt_residents: keep leaves decode's residents in the prompt rows and the next prompt plans them as hits (no record read); forget plans and reads them again" {
    const X = @import("experts.zig");
    const Bx = X.Types(.exl3).Bridge;
    const a = testing.allocator;
    const s = try Synth.write(.{ .hidden = 4096, .inter = 2048, .n_experts = 8 }, "mul1");
    defer s.close();
    var diag: sdk.Diag = .{};
    var bank = try Bank.open(a, testing.io, s.root, 3, synth_implemented, &diag);
    defer bank.deinit();
    var sb = try SDKBank.init(a, &bank);
    defer sb.deinit();
    // The served options of the phase cycle test above (direct reads, phase rows, decode's window 0).
    const prompt_rows = [_]u32{ 4, 4 };
    const decode_rows = [_]u32{ 6, 7 };
    var opts = try X.streamOptionsOf(Bx, &prompt_rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .direct);
    opts.event = null;
    opts.lookahead = null;
    opts.prompt_reads = .stock;
    opts.prompt_ahead = .head;
    opts.prompt_relocate = false;
    opts.ahead_cap = null;
    const wave = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var hits: [2]u32 = .{ 0, 0 };
    var bytes: [2]u64 = .{ 0, 0 };
    for ([_]@import("settings.zig").PromptResidents{ .keep, .forget }, 0..) |residents, i| {
        const st = try Bx.Stream.init(a, &sb, opts);
        defer st.deinit();
        // Request 1 (`Source.beginPrefill` after construction: no decode before it): a prompt of every expert.
        try X.requestStart(st, residents, .construction);
        for (0..2) |l| st.release(try st.serve(@intCast(l), &wave));
        try st.flush();
        // Its decode routes two of each layer's prompt residents and two experts the prompt rows do not hold.
        var dec: [2][4]u16 = undefined;
        for (0..2) |l| {
            var nr: usize = 0;
            var nm: usize = 2;
            for (wave) |e| {
                const resident = st.layers[l].policy.slotOf(e) != null;
                if (resident and nr < 2) {
                    dec[l][nr] = e;
                    nr += 1;
                } else if (!resident and nm < 4) {
                    dec[l][nm] = e;
                    nm += 1;
                }
            }
            try testing.expect(nr == 2 and nm == 4);
        }
        _ = try st.releaseTransient();
        try st.grow(&decode_rows);
        for (0..2) |l| st.release(try st.serve(@intCast(l), &dec[l]));
        try st.flush();
        // The prompt-row slots of the two residents decode used.
        var old: [2][2]u32 = undefined;
        for (0..2) |l| for (0..2) |k| {
            old[l][k] = st.layers[l].policy.slotOf(dec[l][k]).?;
            try testing.expect(old[l][k] < prompt_rows[l]);
        };
        // Request 2: the reverse change (`Source.resumePrefill`) and the request's start (`Source.beginPrefill`).
        _ = try st.shrink(&prompt_rows, switch (residents) {
            .keep => .keep,
            .forget => .forget,
        });
        _ = try st.regrowTransient();
        try X.requestStart(st, residents, .reversed);
        // keep: the prompt rows hold the residents decode found there; forget: none.
        for (0..2) |l| for (dec[l][0..2]) |e| try testing.expectEqual(residents == .keep, st.layers[l].policy.slotOf(e) != null);
        if (residents == .forget) try testing.expectEqual(@as(u64, 0), st.residentCount());
        // Its prompt routes the two kept experts of each layer again, one route each, in the order that puts neither in
        // the slot it held (`forget`'s empty rows take slot 0 then 1): so no load can skip its read on a slot that
        // still holds its record, and every byte forget reads is a record read again.
        const before = st.stats().expert_bytes_read;
        for (0..2) |l| {
            const x = dec[l][0];
            const y = dec[l][1];
            const order: [2]u16 = if (old[l][0] != 0 and old[l][1] != 1) .{ x, y } else .{ y, x };
            for (order) |e| {
                const r = try st.serve(@intCast(l), &.{e});
                hits[i] += r.plan.n_hits;
                st.release(r);
            }
        }
        try st.flush();
        bytes[i] = st.stats().expert_bytes_read - before;
        try testing.expect(!st.failed);
    }
    // keep: the two kept residents of each layer are hits and read nothing; forget: no hit, the four records read again.
    try testing.expectEqual(@as(u32, 4), hits[0]);
    try testing.expectEqual(@as(u32, 0), hits[1]);
    try testing.expectEqual(@as(u64, 0), bytes[0]);
    try testing.expect(bytes[1] > 0);
}

/// The served options of the phase-cycle tests above (direct reads, phase rows, decode's window 0) on `prompt_rows`.
fn cycleOptions(comptime Bx: type, prompt_rows: []const u32) !Bx.Options {
    const X = @import("experts.zig");
    var opts = try X.streamOptionsOf(Bx, prompt_rows, .host, .{ .id = 7, .object = 123 }, .{ .ratio = 0 }, .direct);
    opts.event = null;
    opts.lookahead = null;
    opts.prompt_reads = .stock;
    opts.prompt_ahead = .head;
    opts.prompt_relocate = false;
    opts.ahead_cap = null;
    return opts;
}

test "glm5_next exl3 bank: expert usage: the counts add up across two requests (every prompt routed id, every decode route of an expert) on the host-memory stream; the file round-trips" {
    const X = @import("experts.zig");
    const Bx = X.Types(.exl3).Bridge;
    const usage = sdk_ext.expert.usage;
    const a = testing.allocator;
    const s = try Synth.write(.{ .hidden = 4096, .inter = 2048, .n_experts = 8 }, "mul1");
    defer s.close();
    var diag: sdk.Diag = .{};
    var bank = try Bank.open(a, testing.io, s.root, 3, synth_implemented, &diag);
    defer bank.deinit();
    var sb = try SDKBank.init(a, &bank);
    defer sb.deinit();
    const prompt_rows = [_]u32{ 4, 4 };
    const decode_rows = [_]u32{ 6, 7 };
    const st = try Bx.Stream.init(a, &sb, try cycleOptions(Bx, &prompt_rows));
    defer st.deinit();
    st.countRoutes();
    const wave = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7, 1, 1 };
    const dec = [_]u16{ 6, 1, 7, 3 };
    const counts = try a.alloc(u64, 2 * 8);
    defer a.free(counts);
    @memset(counts, 0);
    var want: [16]u64 = @splat(0);
    for (0..2) |request| {
        if (request == 0) try X.requestStart(st, .keep, .construction) else {
            _ = try st.shrink(&prompt_rows, .keep);
            _ = try st.regrowTransient();
            try X.requestStart(st, .keep, .reversed);
        }
        for (0..2) |l| {
            try st.seedPrefill(@intCast(l), &wave);
            st.release(try st.serve(@intCast(l), &wave));
        }
        try st.flush();
        _ = try st.releaseTransient();
        try st.grow(&decode_rows);
        // Request r decodes r + 1 routes of `dec` per layer.
        for (0..request + 1) |_| for (0..2) |l| st.release(try st.serve(@intCast(l), &dec));
        try st.flush();
        st.addUsage(counts);
        for (0..2) |l| {
            for (wave) |e| want[l * 8 + e] += 1;
            for (dec) |e| want[l * 8 + e] += request + 1;
            // No tally: the decode counts are zeroed by the harvest (the next request's start from 0).
            for (st.layers[l].policy.request_counts) |c| try testing.expectEqual(@as(u32, 0), c);
        }
        try testing.expectEqualSlices(u64, &want, counts);
    }
    // Expert 1 of layer 0: 3 prompt ids a request + 1 + 2 decode routes.
    try testing.expectEqual(@as(u64, 3 + 3 + 1 + 2), counts[1]);
    // The file: written by the recorder, read back as the profile the seed uses.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const path = try std.fs.path.join(a, &.{ dir, usage.file_name });
    defer a.free(path);
    const r = try usage.Recorder.init(a, testing.io, path, "glm5_next", "unit", 2, 8);
    defer r.deinit();
    @memcpy(r.counts, counts);
    try r.endRequest(testing.io, 52, 24);
    var settings: usage.Settings = .{};
    var d = try usage.decide(a, testing.io, &settings, dir, "glm5_next", 2, 8);
    defer d.profile.?.deinit();
    try testing.expectEqualSlices(u64, counts, d.profile.?.counts);
}

test "glm5_next exl3 bank: resident seed: a stream seeded from a usage profile serves its first prompt's seeded experts as hits from their records' bytes (fewer records read than unseeded); the dynamic policy is unchanged; a mismatched profile is refused by name" {
    const X = @import("experts.zig");
    const Bx = X.Types(.exl3).Bridge;
    const usage = sdk_ext.expert.usage;
    const a = testing.allocator;
    const s = try Synth.write(.{ .hidden = 4096, .inter = 2048, .n_experts = 8 }, "mul1");
    defer s.close();
    var diag: sdk.Diag = .{};
    var bank = try Bank.open(a, testing.io, s.root, 3, synth_implemented, &diag);
    defer bank.deinit();
    var sb = try SDKBank.init(a, &bank);
    defer sb.deinit();
    const prompt_rows = [_]u32{ 4, 4 };
    const text = @embedFile("testdata/expert_usage_counts.json");
    try testing.expectError(error.ExpertUsageMismatch, usage.parse(a, text, "glm5_next", 3, 8));
    try testing.expectError(error.ExpertUsageMismatch, usage.parse(a, text, "glm5_next", 2, 16));
    var profile = try usage.parse(a, text, "glm5_next", 2, 8);
    defer profile.deinit();
    const seeded = [2][4]u16{ .{ 1, 3, 5, 7 }, .{ 0, 2, 4, 6 } };
    // The first prompt: every expert of both layers.
    const wave = [_]u16{ 0, 1, 2, 3, 4, 5, 6, 7 };
    var hits: [2]u32 = .{ 0, 0 };
    var bytes: [2]u64 = .{ 0, 0 };
    for ([_]bool{ true, false }, 0..) |seed, i| {
        const st = try Bx.Stream.init(a, &sb, try cycleOptions(Bx, &prompt_rows));
        defer st.deinit();
        if (seed) {
            const Order = struct {
                p: *usage.Profile,
                fn of(o: @This(), layer: u32) []const u16 {
                    return o.p.order(layer, 4);
                }
            };
            const b0 = st.stats().expert_bytes_read;
            try testing.expectEqual(@as(u64, 8), try st.seedResidents(Order{ .p = &profile }, Order.of));
            try testing.expect(st.stats().expert_bytes_read > b0);
            // Each seeded expert a resident as `shrink(.keep)` leaves one: unprotected, recency 0, not pinned.
            for (0..2) |l| {
                const p = &st.layers[l].policy;
                try testing.expectEqual(@as(u32, 4), p.occupancy);
                try testing.expectEqual(@as(usize, 0), p.protected.count());
                for (seeded[l]) |e| {
                    try testing.expect(p.slotOf(e) != null);
                    try testing.expectEqual(@as(u64, 0), p.recency[e]);
                    try testing.expect(!p.isPinned(e));
                }
            }
            // The seed fills empty rows only: a second seed is refused by name.
            try testing.expectError(error.SeedRowsNotEmpty, st.seedResidents(Order{ .p = &profile }, Order.of));
            try X.requestStart(st, .keep, .seeded);
            for (0..2) |l| for (seeded[l]) |e| try testing.expect(st.layers[l].policy.slotOf(e) != null);
        } else try X.requestStart(st, .keep, .construction);
        const before = st.stats().expert_bytes_read;
        for (0..2) |l| {
            try st.seedPrefill(@intCast(l), &wave);
            const r = try st.serve(@intCast(l), &wave);
            hits[i] += r.plan.n_hits;
            // Exact: every id (hit or load) served from its record's bytes.
            try st.expectServed(r, &wave, s.image);
            st.release(r);
        }
        try st.flush();
        bytes[i] = st.stats().expert_bytes_read - before;
        try testing.expect(!st.failed);
    }
    // Seeded: the four seeded experts of each layer are hits; unseeded: none, every record read.
    try testing.expectEqual(@as(u32, 8), hits[0]);
    try testing.expectEqual(@as(u32, 0), hits[1]);
    try testing.expect(bytes[0] < bytes[1]);
}
