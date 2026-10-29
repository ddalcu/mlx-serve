//! Routed experts streamed from an `experts.bin` for a resident trunk (`sdk/weights/streamed.zig` `Streamed`). The
//! source fetches; the MoE math stays the trunk's. `Mxfp4` serves the MXFP4 expert bank format
//! (`mimo_v2/stream/bank.zig`, mlx-stream-mxfp4-expert-bank-v1: the release's FP4 experts as MLX's mxfp4 words and
//! E8M0 scales, six components a record) through the expert stream (`mimo_v2/stream/stream_source.zig`): a call's
//! experts made resident in slot rows, and each routed pair's bank and row.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const ops = @import("ops.zig");
const bank_mod = @import("mimo/bank.zig");
const stream_mod = @import("mimo/stream.zig");
const G = ops.MlxOps;

const T = mlx.mlx_array;

pub const Geometry = bank_mod.Geometry;
pub const Bank = bank_mod.Bank;
/// The rows a wide call borrows beside every layer's slot rows (one set for the stream).
pub const transient_rows: u32 = stream_mod.transient_rows;
/// A call's distinct experts fit one route.
pub const max_route_ids: usize = stream_mod.max_route_ids;
/// Routed pairs a call takes at most (`Fetched`'s buffers).
pub const max_call_ids: usize = 1 << 14;

pub const Proj = struct { w: T, scales: T };
pub const Banks = struct { gate: Proj, up: Proj, down: Proj };
/// One call's slots: per routed pair its bank (`BankKind` order: prompt rows, grown rows, transient rows) and row, and
/// each bank's arrays. Valid until the next `fetch`.
pub const Fetched = struct { kind: []const u8, row: []const u32, banks: [3]?Banks };

pub const Mxfp4 = struct {
    pub const Options = struct {
        /// Slot rows per routed layer (null: every expert, the whole bank resident once read).
        rows: ?[]const u32 = null,
    };

    a: std.mem.Allocator,
    bank: bank_mod.Bank,
    sdk_bank: bank_mod.SDKBank,
    stream: *stream_mod.Stream,
    source: stream_mod.StreamSource,
    g: ops.MlxOps,
    rows: []u32,
    kind: [max_call_ids]u8 = undefined,
    row: [max_call_ids]u32 = undefined,

    /// The bank in `dir` (`expert-manifest.json` + `experts.bin`) streamed on `s`, in the decode phase from the start.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Options) !*Mxfp4 {
        var bank = try bank_mod.Bank.open(a, io, dir);
        errdefer bank.deinit();
        const geo = bank.geometry;
        const rows = try a.alloc(u32, geo.num_moe_layers);
        errdefer a.free(rows);
        if (opt.rows) |r| {
            if (r.len != rows.len) return error.StreamedRows;
            for (r) |n| if (n == 0 or n > geo.num_experts) return error.StreamedRows;
            @memcpy(rows, r);
        } else @memset(rows, geo.num_experts);
        const self = try a.create(Mxfp4);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .bank = bank, .sdk_bank = undefined, .stream = undefined, .source = undefined, .g = undefined, .rows = rows };
        self.sdk_bank = try bank_mod.SDKBank.init(a, &self.bank);
        errdefer self.sdk_bank.deinit();
        var opts = stream_mod.options(rows, .{ .mlx = s });
        opts.lookahead = stream_mod.lookahead;
        self.stream = try stream_mod.Stream.init(a, &self.sdk_bank, opts);
        errdefer self.stream.deinit();
        self.source = stream_mod.StreamSource.init(self.stream);
        self.g = try ops.MlxOps.init(a, s);
        errdefer self.g.deinit();
        try self.source.grow(rows);
        return self;
    }

    pub fn deinit(self: *Mxfp4) void {
        _ = mlx.mlx_synchronize(self.g.s);
        self.g.deinit();
        self.stream.deinit();
        self.sdk_bank.deinit();
        self.bank.deinit();
        self.a.free(self.rows);
        self.a.destroy(self);
    }

    /// Model layer `layer`'s routed pairs `ids` (the router's flat order) made resident: one route over their distinct
    /// experts, its reads waited on the host. The released routes are unpinned first (`ids` came from an eval that
    /// consumed them); this call's slots stay pinned until the next `fetch` or `settle`.
    pub fn fetch(self: *Mxfp4, layer: u32, ids: []const u16, scores: []const f32) !Fetched {
        if (ids.len > max_call_ids) return error.StreamedCallTooWide;
        try self.source.flush();
        var distinct: [max_route_ids]u16 = undefined;
        var at: [1 << 16]u8 = undefined;
        var n: usize = 0;
        for (ids) |e| {
            if (std.mem.indexOfScalar(u16, distinct[0..n], e) != null) continue;
            if (n == max_route_ids) return error.StreamedCallTooWide;
            distinct[n] = e;
            n += 1;
        }
        const call = try self.source.route(layer, distinct[0..n], scores);
        const sv = self.source.served(call);
        for (0..sv.n_parts) |p| {
            try self.source.waitGu(call, @intCast(p));
            try self.source.waitDown(call, @intCast(p));
        }
        for (distinct[0..n], 0..) |e, i| at[e] = @intCast(i);
        for (ids, 0..) |e, i| {
            const ref = sv.refs[at[e]];
            self.kind[i] = @intFromEnum(ref.bank);
            self.row[i] = ref.row;
        }
        self.source.release(call);
        var banks: [3]?Banks = @splat(null);
        for ([_]stream_mod.BankKind{ .base, .ext, .transient }, 0..) |k, b| if (try self.source.bankArrays(&self.g, layer, k)) |ba| {
            banks[b] = .{
                .gate = .{ .w = ba.gate.w, .scales = ba.gate.scales.? },
                .up = .{ .w = ba.up.w, .scales = ba.up.scales.? },
                .down = .{ .w = ba.down.w, .scales = ba.down.scales.? },
            };
        };
        return .{ .kind = self.kind[0..ids.len], .row = self.row[0..ids.len], .banks = banks };
    }

    /// After the eval that consumed the last call's slots.
    pub fn settle(self: *Mxfp4) !void {
        try self.source.flush();
    }

    pub fn stats(self: *Mxfp4) stream_mod.Stats {
        return self.source.stats();
    }
};

/// The bank in `dir` (MXFP4 records).
pub const Source = union(enum) {
    mxfp4: *Mxfp4,

    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, s: mlx.mlx_stream, opt: Mxfp4.Options) !Source {
        return .{ .mxfp4 = try Mxfp4.open(a, io, dir, s, opt) };
    }

    pub fn deinit(self: Source) void {
        switch (self) {
            inline else => |x| x.deinit(),
        }
    }

    pub fn settle(self: Source) !void {
        switch (self) {
            inline else => |x| try x.settle(),
        }
    }
};

/// Writes the bank `Mxfp4` reads: `experts.bin` (`bytes`, every routed layer's records in order) and its manifest.
pub fn writeBank(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, geo: Geometry, bytes: []const u8, repository: []const u8, revision: []const u8) !void {
    if (bytes.len != geo.total_bytes) return error.StreamedBankBytes;
    try dir.writeFile(io, .{ .sub_path = "experts.bin", .data = bytes });
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"format\":\"{s}\",\"data_file\":\"experts.bin\",\"hidden_size\":{d},\"intermediate_size\":{d},\"num_experts\":{d},\"first_moe_layer\":{d},\"num_moe_layers\":{d},\"record_bytes\":{d},\"gate_up_bytes\":{d},", .{ bank_mod.format, geo.hidden_size, geo.intermediate_size, geo.num_experts, geo.first_moe_layer, geo.num_moe_layers, geo.record_bytes, geo.gate_up_bytes });
    try w.print("\"quantization\":{{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":32}},\"source\":{{\"repository\":\"{s}\",\"revision\":\"{s}\"}},\"segments\":[", .{ repository, revision });
    for (geo.segments, 0..) |seg, i| {
        const c: bank_mod.Component = @fromBackingInt(@intCast(i));
        try w.print("{s}{{\"name\":\"{s}\",\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"offset\":{d},\"length\":{d}}}", .{ if (i > 0) "," else "", c.name(), @tagName(seg.dtype), seg.shape[0], seg.shape[1], seg.offset, seg.length });
    }
    try w.writeAll("],\"records\":[");
    const count = @as(usize, geo.num_experts) * geo.num_moe_layers;
    for (0..count) |i| {
        const rec = bytes[i * geo.record_bytes ..][0..geo.record_bytes];
        try w.print("{s}{{\"layer\":{d},\"expert\":{d},\"offset\":{d},\"length\":{d},\"sha256\":\"{s}\"}}", .{ if (i > 0) "," else "", geo.first_moe_layer + i / geo.num_experts, i % geo.num_experts, i * geo.record_bytes, geo.record_bytes, &sha256(rec) });
    }
    try w.print("],\"output_files\":{{\"experts.bin\":{{\"size\":{d},\"sha256\":\"{s}\"}}}}}}", .{ bytes.len, &sha256(bytes) });
    try dir.writeFile(io, .{ .sub_path = "expert-manifest.json", .data = out.written() });
}

fn sha256(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

