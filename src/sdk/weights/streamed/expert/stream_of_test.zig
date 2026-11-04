//! The bank contract's second shape: a MiMo-like MXFP4 bank module (six components, gate/up range four, every segment a
//! 16 KiB multiple, one experts.bin) drives the same stream code on host rows. No MLX array, no device.

const std = @import("std");
const sdk = @import("sdk");
const sdk_ext = @import("../sdk_ext.zig");
const mlx = @import("sdk").mlx;
const StreamOf = sdk_ext.expert.stream.StreamOf;

const testing = std.testing;

/// A bank module in MiMo's record layout (gate / up / down weight U32 then scales U8), at a test's size.
const MxBank = struct {
    pub const n_components = 6;
    pub const gu_components = 4;
    pub const Records = sdk_ext.expert.Records(n_components, gu_components);
    pub const Component = enum(u8) { gate_weight, gate_scales, up_weight, up_scales, down_weight, down_scales };
    pub const Dtype = enum { U32, U8 };
    pub const Segment = struct { offset: u64, length: u64, dtype: Dtype, shape: [3]u64, rank: u8 };
    pub const Layer = struct { logical_bytes: u64, segments: [n_components]Segment };
    pub const BankArrays = [n_components]mlx.mlx_array;
    pub const routed_top_k = 8;

    pub fn mlxDtype(d: Dtype) mlx.mlx_dtype {
        return switch (d) {
            .U32 => .uint32,
            .U8 => .uint8,
        };
    }

    pub fn bankArraysOf(x: [n_components]mlx.mlx_array) BankArrays {
        return x;
    }

    pub const Spans = struct { gu_offset: u64, down_offset: u64 };

    pub const Bank = struct {
        layers: []Layer,
        n_experts: u32,
        sidecar: sdk_ext.expert.UncachedFd,
        record_bytes: u64,

        pub fn recordOffset(self: *const Bank, layer: u32, expert: u32) u64 {
            return (@as(u64, layer) * self.n_experts + expert) * self.record_bytes;
        }

        pub fn spans(self: *const Bank, layer: u32, expert: u32) Spans {
            const off = self.recordOffset(layer, expert);
            return .{ .gu_offset = off, .down_offset = off + self.layers[layer].segments[gu_components].offset };
        }
    };

    /// MiMo's segment table at hidden `h`, intermediate `i` (weights 4 bits packed in U32, scales one U8 per 32).
    fn layerOf(h: u64, i: u64) Layer {
        var l: Layer = .{ .logical_bytes = 0, .segments = undefined };
        var off: u64 = 0;
        for (0..3) |p| {
            const in = if (p == 2) i else h;
            const out = if (p == 2) h else i;
            l.segments[p * 2] = .{ .offset = off, .length = out * in / 2, .dtype = .U32, .shape = .{ out, in / 8, 0 }, .rank = 2 };
            off += out * in / 2;
            l.segments[p * 2 + 1] = .{ .offset = off, .length = out * in / 32, .dtype = .U8, .shape = .{ out, in / 32, 0 }, .rank = 2 };
            off += out * in / 32;
        }
        l.logical_bytes = off;
        return l;
    }
};

const MxStream = StreamOf(MxBank, false).Stream;

test "dsv41 bank contract: a MiMo-shaped MXFP4 bank (6 components, gate/up 4) streams through the same code, every slot its record" {
    // h 256, i 128: weights 16 KiB, scales 1 KiB per projection; record 52 KiB (MiMo: 13,369,344 B at h 4096, i 2048).
    var layers = [_]MxBank.Layer{ MxBank.layerOf(256, 128), MxBank.layerOf(256, 128) };
    const rec = layers[0].logical_bytes;
    try testing.expectEqual(@as(u64, 3 * (16384 + 1024)), rec);
    const n_experts: u32 = 24;
    const total = rec * n_experts * layers.len;
    const image = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(image);
    for (image, 0..) |*b, k| b.* = @truncate(k *% 2654435761 >> 9);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "experts.bin", .data = image });
    var root: [512]u8 = undefined;
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/experts.bin", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    defer testing.allocator.free(path);
    const fd = try sdk_ext.expert.openUncached(path.ptr, null);
    defer fd.close();
    const bank: MxBank.Bank = .{ .layers = &layers, .n_experts = n_experts, .sidecar = fd, .record_bytes = rec };

    const s = try MxStream.init(testing.allocator, &bank, .{ .rows = &.{ 4, 3 }, .max_route_ids = 8, .transient_rows = 8, .pool = .{ .workers = 2, .staging_bytes = 65536, .tickets = 256 } });
    defer s.deinit();
    var rng = std.Random.DefaultPrng.init(11);
    var ids: [8]u16 = undefined;
    for (0..40) |step| {
        const l: u32 = @intCast(step % 2);
        const n = 1 + rng.random().uintLessThan(usize, ids.len);
        var k: usize = 0;
        while (k < n) {
            const e = rng.random().uintLessThan(u16, @intCast(n_experts));
            if (std.mem.indexOfScalar(u16, ids[0..k], e) == null) {
                ids[k] = e;
                k += 1;
            }
        }
        const r = try s.serve(l, ids[0..n]);
        try s.expectServed(r, ids[0..n], image);
        s.release(r);
    }
}

test "dsv41 bank contract: a top-8 arch's lookahead measures tau from each row's eighth score, a top-6 arch's from its sixth" {
    const a = testing.allocator;
    // One row over 16 experts, scores 15 - e: the 6th score is 10, the 8th is 8. With tau 1 and k 10, a top-6 selector keeps
    // scores >= 9 (experts 0..6), a top-8 selector scores >= 7 (experts 0..8); none is resident.
    var row: [16]f32 = undefined;
    for (&row, 0..) |*v, e| v.* = 15 - @as(f32, @floatFromInt(e));
    var none = try sdk_ext.expert.policy.LayerPolicy.init(a, 16, 0);
    defer none.deinit(a);
    var out: [4]u16 = undefined;
    var s6 = try sdk_ext.expert.lookahead.SelectorOf(6).init(a, 16, 10, 1.0, 4);
    defer s6.deinit(a);
    var s8 = try sdk_ext.expert.lookahead.SelectorOf(8).init(a, 16, 10, 1.0, 4);
    defer s8.deinit(a);
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, s6.select(&row, &none, &out));
    try testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, s8.select(&row, &none, &out));
    // The candidate counts differ by the threshold: read through a budget wide enough to see the cut.
    var wide: [12]u16 = undefined;
    try testing.expectEqual(@as(usize, 7), s6.select(&row, &none, &wide).len);
    try testing.expectEqual(@as(usize, 9), s8.select(&row, &none, &wide).len);
    // A top-8 arch cannot keep fewer than eight candidates per row.
    try testing.expectError(error.InvalidSelector, sdk_ext.expert.lookahead.SelectorOf(8).init(a, 16, 7, 1.0, 4));
}

test "dsv41 bank contract: a bank whose gate/up span exceeds the pool's 9 MiB staging pre-reads once the staging follows the bank" {
    // GLM-5.3's affine record shape at one layer: the gate/up span 14,155,776 B, the down span 7,077,888 B (each
    // segment as its own one-row geometry; only the lengths matter here).
    var l: MxBank.Layer = .{ .logical_bytes = 0, .segments = undefined };
    const lens = [_]u64{ 6_291_456, 786_432, 6_291_456, 786_432, 6_291_456, 786_432 };
    var off: u64 = 0;
    for (&l.segments, lens, 0..) |*sg, n, c| {
        sg.* = .{ .offset = off, .length = n, .dtype = if (c % 2 == 0) .U32 else .U8, .shape = .{ 1, n / (if (c % 2 == 0) @as(u64, 4) else 1), 0 }, .rank = 2 };
        off += n;
    }
    l.logical_bytes = off;
    var layers = [_]MxBank.Layer{l};
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "experts.bin", .data = "" });
    var root: [512]u8 = undefined;
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/experts.bin", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    defer testing.allocator.free(path);
    const fd = try sdk_ext.expert.openUncached(path.ptr, null);
    defer fd.close();
    const bank: MxBank.Bank = .{ .layers = &layers, .n_experts = 8, .sidecar = fd, .record_bytes = std.mem.alignForward(u64, off, 4096) };
    const page = std.heap.pageSize();
    try testing.expectEqual(std.mem.alignForward(u64, 14_155_776, page) + page, StreamOf(MxBank, false).stagingBytes(&bank));
    const opt: StreamOf(MxBank, false).Options = .{ .rows = &.{1}, .max_route_ids = 1, .transient_rows = 1, .lookahead = .{ .budget = 1 }, .pool = .{ .workers = 1, .tickets = 256 } };
    // The pool's default staging (9 MiB) cannot hold the gate/up span: the pre-read is refused at construction.
    try testing.expectError(error.InvalidOptions, MxStream.init(testing.allocator, &bank, opt));
    var sized = opt;
    sized.staging_from_bank = true;
    const s = try MxStream.init(testing.allocator, &bank, sized);
    defer s.deinit();
    try testing.expectEqual(StreamOf(MxBank, false).stagingBytes(&bank), s.pool.staging.len);
}

/// MxBank's records as GLM-5.3's EXL3 bank holds them: a routed id's `minis` records adjacent in the file at
/// `record_bytes`, and in its slot's rows (`Layer.minis`).
const MinisBank = struct {
    pub const n_components = MxBank.n_components;
    pub const gu_components = MxBank.gu_components;
    pub const Records = MxBank.Records;
    pub const Component = MxBank.Component;
    pub const BankArrays = MxBank.BankArrays;
    pub const routed_top_k = MxBank.routed_top_k;
    pub const mlxDtype = MxBank.mlxDtype;
    pub const bankArraysOf = MxBank.bankArraysOf;
    pub const Layer = struct { logical_bytes: u64, record_bytes: u64, minis: u32, segments: [n_components]MxBank.Segment };

    pub const Bank = struct {
        layers: []Layer,
        n_experts: u32,
        sidecar: sdk_ext.expert.UncachedFd,

        /// The id's first record; its others follow at `record_bytes`.
        pub fn recordOffset(self: *const Bank, layer: u32, expert: u32) u64 {
            const l = &self.layers[layer];
            return (@as(u64, layer) * self.n_experts + expert) * l.minis * l.record_bytes;
        }

        pub fn spans(self: *const Bank, layer: u32, expert: u32) MxBank.Spans {
            const off = self.recordOffset(layer, expert);
            return .{ .gu_offset = off, .down_offset = off + self.layers[layer].segments[gu_components].offset };
        }
    };
};

const MinisOf = StreamOf(MinisBank, false);

test "dsv41 bank contract: host slot rows of a bank of 4 records per routed id: each bank holds all its rows, the last persistent and the last transient row written and read back" {
    const a = testing.allocator;
    const minis = 4;
    const mx = MxBank.layerOf(256, 128);
    const geom: MinisBank.Layer = .{ .logical_bytes = mx.logical_bytes, .record_bytes = std.mem.alignForward(u64, mx.logical_bytes, 4096), .minis = minis, .segments = mx.segments };
    var layers = [_]MinisBank.Layer{ geom, geom };
    const n_experts: u32 = 12;
    const image = try a.alloc(u8, layers.len * n_experts * minis * geom.record_bytes);
    defer a.free(image);
    for (image, 0..) |*b, k| b.* = @truncate(k *% 2654435761 >> 9);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "experts.bin", .data = image });
    var root: [512]u8 = undefined;
    const path = try std.fmt.allocPrintSentinel(a, "{s}/experts.bin", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    defer a.free(path);
    const fd = try sdk_ext.expert.openUncached(path.ptr, null);
    defer fd.close();
    const bank: MinisBank.Bank = .{ .layers = &layers, .n_experts = n_experts, .sidecar = fd };

    // Layer 0 has 3 persistent rows; layer 1 has none, so its misses take the transient rows.
    const s = try MinisOf.Stream.init(a, &bank, .{ .rows = &.{ 3, 0 }, .max_route_ids = 4, .transient_rows = 4, .records_per_part = 2, .staging_from_bank = true, .pool = .{ .workers = 2, .tickets = 256 } });
    defer s.deinit();
    // A row is a segment of each of the 4 records, back to back: each component's bank is its rows times that.
    for ([_]*const MinisOf.Rows{ &s.layers[0].base, &s.transient }) |rows| {
        for (rows.backing.host.banks, geom.segments) |b, seg| try testing.expectEqual(rows.rows * minis * seg.length, b.len);
    }
    // Layer 0's three misses fill its rows 0, 1 and 2; layer 1's four take transient rows 0 to 3.
    const Case = struct { layer: u32, ids: []const u16, last: u32 };
    for ([_]Case{ .{ .layer = 0, .ids = &.{ 7, 2, 10 }, .last = 2 }, .{ .layer = 1, .ids = &.{ 5, 0, 11, 3 }, .last = 3 } }) |x| {
        const r = try s.serve(x.layer, x.ids);
        try testing.expect(std.mem.indexOfScalar(u32, r.plan.slotsOf(), x.last) != null);
        try s.expectServed(r, x.ids, image);
        s.release(r);
    }
    try s.flush();
}
