//! The Qwen family's load over the engine's loaded tensors (`LoadCtx.weights`): each projection read through the
//! pack's quant format, quantized row groups joined into one buffer exactly as the engine's loader joins them, and
//! every handle the model keeps owned in one list. Shared by the qwen3, qwen3_5 and qwen3_asr plugins.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const bases = @import("../../sdk/bases/bases.zig");
const formats = @import("../../sdk/quant/formats.zig");
const native = @import("../../sdk/quant/native.zig");
const hadamard = @import("../../sdk/quant/hadamard/format.zig");
const fuseRowGroup = @import("../../sdk/blocks/linear/fused_rows.zig").fuseRowGroup;
const daq = @import("../../sdk/blocks/attention/decode_attn_quant.zig");
const log = @import("../../sdk/util/log.zig");

const Array = mlx.mlx_array;
pub const Linear = bases.blocks.linear.Linear;
pub const Joined = bases.blocks.linear.Joined;

pub const gpa = std.heap.c_allocator;

/// The handles a loaded model holds, freed together.
pub const Owned = struct {
    list: std.ArrayList(Array) = .empty,

    pub fn own(o: *Owned, a: Array) !void {
        if (a.ctx != null) try o.list.append(gpa, a);
    }

    pub fn linear(o: *Owned, l: Linear) !void {
        for ([_]Array{ l.w, l.s, l.b, l.bias }) |a| try o.own(a);
    }

    pub fn deinit(o: *Owned) void {
        for (o.list.items) |a| _ = mlx.mlx_array_free(a);
        o.list.deinit(gpa);
    }
};

fn ctxArr(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// Where the tensors come from during one load.
pub const Reader = struct {
    src: *const quant_abi.Source,
    fmt: *const quant_abi.QuantTable,
    qctx: quant_abi.LoadCtx,
    prefix: []const u8,
    s: mlx.mlx_stream,
    /// Every rotated projection read so far has biases exactly -scales (`hadamard.NEG_SCALE`).
    neg_scale: bool = true,

    /// Opens the pack's quant format over `src` (`block_json` the format's view of the quantization block).
    pub fn open(r: *Reader, src: *const quant_abi.Source, block_json: []const u8, prefix: []const u8, s: mlx.mlx_stream) !void {
        return r.openIn(src, block_json, "{}", prefix, s);
    }

    /// `open`, the format claimed over `config_json` too (a Hadamard pack's `modules`).
    pub fn openIn(r: *Reader, src: *const quant_abi.Source, block_json: []const u8, config_json: []const u8, prefix: []const u8, s: mlx.mlx_stream) !void {
        const fmt = (formats.claimant(block_json, config_json) catch null) orelse &native.affine;
        r.* = .{
            .src = src,
            .fmt = fmt,
            .qctx = .{ .block_json = abi.Str.of(block_json), .config_json = abi.Str.of(config_json), .source = src, .stream = .{ .ctx = s.ctx } },
            .prefix = prefix,
            .s = s,
        };
        if (!fmt.open(&r.qctx).isOk()) return error.QuantFormatRefused;
    }

    pub fn close(r: *Reader) void {
        r.fmt.close(&r.qctx);
    }

    /// `<prefix>.<fmt>`.
    pub fn name(r: *const Reader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}." ++ fmt, .{r.prefix} ++ args);
    }

    pub fn has(r: *const Reader, full: []const u8) bool {
        return r.src.tensor(full).ctx != null;
    }

    /// A new handle on the source's tensor `full`, empty when the pack has none.
    pub fn share(r: *const Reader, full: []const u8) !Array {
        const a = ctxArr(r.src.tensor(full));
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

    /// The projection `base` through the pack's quant format (`in_dim` pins its quant params), unowned; a rotated
    /// one carries its own sign vector (`signs`, unowned) until the model pools it.
    pub fn proj(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var w: quant_abi.Weight = .{};
        const v = r.fmt.load_tensor(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        if (w.layout == @backingInt(native.Layout.fp8_rows)) {
            r.fmt.release(&w);
            return error.RowScaledFp8Unsupported;
        }
        if (w.side.ctx != null) r.neg_scale = r.neg_scale and w.flags & hadamard.NEG_SCALE != 0;
        return .{ .w = ctxArr(w.w), .s = ctxArr(w.scales), .b = ctxArr(w.biases), .bits = w.bits, .group_size = w.group_size, .mode = native.modeOf(&w), .signs = ctxArr(w.side) };
    }

    /// A routed-expert bank `base` (`[experts, out, in]`) through the pack's quant format, unowned.
    pub fn bank(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var w: quant_abi.Weight = .{};
        const v = r.fmt.load_bank(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        return .{ .w = ctxArr(w.w), .s = ctxArr(w.scales), .b = ctxArr(w.biases), .bits = w.bits, .group_size = w.group_size, .mode = native.modeOf(&w) };
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

    /// The head as the engine finds it: under the decoder's own prefix, then at the root, then the tied embedding.
    pub fn head(r: *Reader, o: *Owned, embed: Linear, hidden: u32, tied: bool) !Linear {
        var nb: [160]u8 = undefined;
        const nested = if (std.mem.eql(u8, r.prefix, "language_model.model")) "language_model" else r.prefix;
        for ([_][]const u8{ try std.fmt.bufPrint(&nb, "{s}.lm_head", .{nested}), "lm_head" }) |base| {
            var kb: [176]u8 = undefined;
            if (r.src.tensor(try std.fmt.bufPrint(&kb, "{s}.weight", .{base})).ctx == null) continue;
            const h = try r.tableAt(base, hidden);
            try o.linear(h);
            return h;
        }
        if (!tied) return error.MissingWeight;
        return embed;
    }
};

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

/// `weight_prefix`: the config's `preferred` when the checkpoint has its embedding there, else the first layout
/// mlx-lm writes that it does have (the engine's `resolveWeightPrefix`).
pub fn probePrefix(src: *const quant_abi.Source, preferred: []const u8) ?[]const u8 {
    for ([_][]const u8{ preferred, "language_model.model", "model.language_model", "model" }) |p| {
        var nb: [96]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "{s}.embed_tokens.weight", .{p}) catch continue;
        if (src.tensor(key).ctx != null) return p;
    }
    return null;
}

/// The decode-only requantized copy of an attention projection (`--decode-attn-quant`), owned by `o`; empty where
/// the engine builds none (the flag off, a quantized weight, an ineligible shape).
/// copied_from: sdk/blocks/attention/proj.zig (`attnDqFor`; built at load, not at the first decode step).
pub fn sideCopy(o: *Owned, l: Linear, layer: u32, layers: u32, s: mlx.mlx_stream) !Linear {
    if (!daq.decodeAttnQuantEnabled() or l.s.ctx != null) return .{};
    if (!daq.decodeAttnQuantEligible(mlx.getShape(l.w), mlx.mlx_array_dtype(l.w), false)) return .{};
    const copy = (try daq.buildAttnDqCopy(s, l.w, daq.attnDqUseNvfp4(layer, layers))) orelse return .{};
    const out: Linear = .{ .w = copy.w, .s = copy.s, .b = copy.b, .bits = copy.bits, .group_size = copy.group_size, .mode = copy.mode };
    try o.linear(out);
    if (!daq.attn_dq_engaged) {
        daq.attn_dq_engaged = true;
        log.info("[decode-attn-quant] engaged: dense attention projections -> int8 g32 side copies (decode-only, lossy; --no-decode-attn-quant restores dense)\n", .{});
    }
    return out;
}

/// A drafter's view of the trunk's embedding table or head (`SpecHooks.trunk_weight`), borrowed.
pub fn trunkWeight(l: *const Linear, out: *quant_abi.Weight) abi.Verdict {
    out.* = .{ .w = .{ .ctx = l.w.ctx }, .scales = .{ .ctx = l.s.ctx }, .biases = .{ .ctx = l.b.ctx }, .bits = @intCast(l.bits), .group_size = l.group_size };
    if (l.rht) |reg| if (l.signs.ctx != null) {
        out.side = .{ .ctx = l.signs.ctx };
        out.block = @intCast(reg.block);
        if (reg.bias_is_neg_scale) out.flags = hadamard.NEG_SCALE;
    };
    return .ok;
}

/// A table load's verdict from a Zig error (its name is static text).
pub fn failed(e: anyerror) abi.Verdict {
    return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
}
