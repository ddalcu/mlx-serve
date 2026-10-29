//! The dense family's load helpers over the engine's loaded tensors (`LoadCtx.weights`): a reader that resolves each
//! projection through the pack's quant format, the arrays a model owns, quantized row joins (q|k|v, gate|up) built
//! as the engine's loader built them, the weight prefix probe and the head lookup. Shared by llama and the plugins
//! that compose the same decoder (k2_horizon, spark2_5, laguna, flint, kolibri1, muse_glimmer).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const formats = @import("../../sdk/quant/formats.zig");
const native = @import("../../sdk/quant/native.zig");
const gguf_format = @import("../../sdk/quant/gguf/format.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;
const transposeBf16Weight = @import("../../mlx/array.zig").transposeBf16Weight;
const fuseRowGroup = @import("../../sdk/blocks/linear/fused_rows.zig").fuseRowGroup;

const gpa = std.heap.c_allocator;
const Array = mlx.mlx_array;
const Linear = bases.blocks.linear.Linear;
const Joined = bases.blocks.linear.Joined;
const Bank = bases.blocks.moe.Bank;

/// Every array handle a loaded model holds, freed together.
pub const Owned = struct {
    list: std.ArrayList(Array) = .empty,

    pub fn own(o: *Owned, a: Array) !void {
        if (a.ctx != null) try o.list.append(gpa, a);
    }

    pub fn linear(o: *Owned, l: Linear) !void {
        for ([_]Array{ l.w, l.s, l.b, l.bias }) |a| try o.own(a);
    }

    pub fn release(o: *Owned) void {
        for (o.list.items) |a| _ = mlx.mlx_array_free(a);
        o.list.deinit(gpa);
    }
};

/// Where the tensors come from during one load.
pub const Reader = struct {
    src: *const quant_abi.Source,
    fmt: *const quant_abi.QuantTable,
    qctx: quant_abi.LoadCtx,
    prefix: []const u8,
    s: mlx.mlx_stream,

    /// Opens the pack's quant format over `ctx.weights` (`block_json` = the `quantization` block, "{}" = dense) and
    /// probes the weight prefix among `prefixes` by `<prefix>.<probe>`. Close with `close`.
    pub fn open(ctx: *const abi.LoadCtx, block_json: []const u8, prefixes: []const []const u8, probe: []const u8) !Reader {
        const src = ctx.weights orelse return error.NoWeightSource;
        const fmt = (formats.claimant(block_json, "{}") catch null) orelse &native.affine;
        var r: Reader = .{
            .src = src,
            .fmt = fmt,
            .qctx = .{ .block_json = abi.Str.of(block_json), .source = src, .stream = ctx.stream },
            .prefix = probePrefix(src, prefixes, probe) orelse return error.MissingWeight,
            .s = .{ .ctx = ctx.stream.ctx },
        };
        if (!fmt.open(&r.qctx).isOk()) return error.QuantFormatRefused;
        return r;
    }

    pub fn close(r: *Reader) void {
        r.fmt.close(&r.qctx);
    }

    /// `<prefix>.<fmt>` in `buf`.
    /// The format by name: a table's address is the compile unit's own.
    fn isGguf(r: *const Reader) bool {
        return std.mem.eql(u8, r.fmt.name.slice(), gguf_format.table.name.slice());
    }

    pub fn name(r: *const Reader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}." ++ fmt, .{r.prefix} ++ args);
    }

    /// Whether the source holds `full`.
    pub fn has(r: *const Reader, full: []const u8) bool {
        return r.src.tensor(full).ctx != null;
    }

    /// A new handle on the source's tensor `full`, empty when the pack has none.
    pub fn share(r: *const Reader, full: []const u8) !Array {
        const a: Array = .{ .ctx = r.src.tensor(full).ctx };
        if (a.ctx == null) return a;
        var h = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_array_set(&h, a));
        return h;
    }

    pub fn need(r: *const Reader, full: []const u8) !Array {
        const a = try r.share(full);
        if (a.ctx == null) return error.MissingWeight;
        return a;
    }

    fn weight(r: *Reader, base: []const u8, in_dim: u32) !quant_abi.Weight {
        var w: quant_abi.Weight = .{};
        // A GGUF's float tensors load as bf16 beside its block-quantized ones: one without a sentinel is dense.
        if (r.isGguf()) {
            var nb: [256]u8 = undefined;
            if (!r.has(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}))) {
                const raw = r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
                if (raw.ctx == null) return error.MissingWeight;
                w.w = .{ .ctx = (try transposeBf16Weight(.{ .ctx = raw.ctx }, r.s)).ctx };
                w.layout = @backingInt(native.Layout.dense);
                return w;
            }
        }
        const v = r.fmt.load_tensor(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        if (w.layout == @backingInt(native.Layout.fp8_rows)) {
            r.fmt.release(&w);
            return error.RowScaledFp8Unsupported;
        }
        return w;
    }

    /// The projection `base` through the pack's quant format (`in_dim` pins its quant params), unowned.
    pub fn proj(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        const w = try r.weight(base, in_dim);
        return .{ .w = arr(w.w), .s = arr(w.scales), .b = arr(w.biases), .bits = w.bits, .group_size = w.group_size, .mode = r.modeOf(&w) };
    }

    /// A GGUF weight's layout is its block type, read by the kernels off the sentinel; never an MLX mode.
    fn modeOf(r: *const Reader, w: *const quant_abi.Weight) QuantMode {
        return if (r.isGguf() and w.scales.ctx != null) .gguf else native.modeOf(w);
    }

    /// Routed experts stacked under `base` (`[E, out, in]`; dense ones `[E, in, out]`), unowned. A dense bank in a
    /// quantized pack is refused: the engine gives it the pack's scheme, which no kernel here reads that way.
    pub fn bank(r: *Reader, base: []const u8, in_dim: u32) !Bank {
        const w = try r.weight(base, in_dim);
        if (w.scales.ctx == null) {
            if (std.mem.eql(u8, r.qctx.block_json.slice(), "{}")) return .{ .w = arr(w.w), .qp = .{ .bits = 0, .group_size = 64 } };
            var x = w;
            r.fmt.release(&x);
            return error.DenseExpertBankUnsupported;
        }
        return .{ .w = arr(w.w), .s = arr(w.scales), .b = arr(w.biases), .qp = .{ .bits = w.bits, .group_size = w.group_size, .mode = r.modeOf(&w) } };
    }

    /// A table kept in its checkpoint layout `[out, in]` (the embedding, the head): dense by its dtype, as the engine
    /// decides it, else through the format.
    pub fn tableAt(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var nb: [256]u8 = undefined;
        const w = r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
        if (w.ctx == null) return error.MissingWeight;
        switch (mlx.mlx_array_dtype(.{ .ctx = w.ctx })) {
            .bfloat16, .float16, .float32 => return .{ .w = try r.need(nb[0 .. base.len + 7]) },
            else => {},
        }
        if (r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.scales", .{base})).ctx == null) return error.MissingWeight;
        return r.proj(base, in_dim);
    }
};

pub fn arr(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// Joins quantized projections reading one input into one buffer (rows) when they share a packed geometry, rewriting
/// each to a row view of it and freeing its own copy; empty when they do not.
pub fn join(o: *Owned, parts: []const *Linear, s: mlx.mlx_stream) !Joined {
    var triples: [4][3]*Array = undefined;
    var originals: [4][3]Array = undefined;
    for (parts, 0..) |p, i| {
        triples[i] = .{ &p.w, &p.s, &p.b };
        originals[i] = .{ p.w, p.s, p.b };
    }
    const fused = (try fuseRowGroup(triples[0..parts.len], s)) orelse return .{};
    for (originals[0..parts.len]) |orig| for (orig) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Array{ fused.w, fused.s, fused.b }) |a| try o.own(a);
    const q = parts[0];
    return .{ .lin = .{ .w = fused.w, .s = fused.s, .b = fused.b, .bits = q.bits, .group_size = q.group_size, .mode = q.mode }, .widths = fused.widths };
}

/// The first of `prefixes` under which the source holds `<prefix>.<probe>`.
pub fn probePrefix(src: *const quant_abi.Source, prefixes: []const []const u8, probe: []const u8) ?[]const u8 {
    for (prefixes) |p| {
        var nb: [160]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "{s}.{s}", .{ p, probe }) catch continue;
        if (src.tensor(key).ctx != null) return p;
    }
    return null;
}

/// The head as the engine finds it: under the decoder's own prefix (`language_model` for a nested trunk), then at the
/// root, then the tied embedding.
pub fn headOf(r: *Reader, o: *Owned, embed: Linear, hidden: u32, tied: bool) !Linear {
    var nb: [160]u8 = undefined;
    const nested = if (std.mem.eql(u8, r.prefix, "language_model.model")) "language_model" else r.prefix;
    for ([_][]const u8{ try std.fmt.bufPrint(&nb, "{s}.lm_head", .{nested}), "lm_head" }) |base| {
        var kb: [176]u8 = undefined;
        if (r.src.tensor(try std.fmt.bufPrint(&kb, "{s}.weight", .{base})).ctx == null) continue;
        const head = try r.tableAt(base, hidden);
        try o.linear(head);
        return head;
    }
    if (!tied) return error.MissingWeight;
    return embed;
}

/// A 0-d bf16 scalar (the legacy `bf16Scalar`), owned by `o`.
pub fn bf16Scalar(o: *Owned, v: f32, s: mlx.mlx_stream) !Array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    try o.own(out);
    return out;
}
