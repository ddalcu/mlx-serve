//! GGUF block types that ARE an affine code: every group of weights (32; 64 for Q2_0) shares one (scale, bias) and
//! w = scale * q + bias for an unsigned q. Such a tensor moves into MLX's affine layout (`mx.quantize` packing) without
//! requantizing: the codes stay, only the scales round to bf16. The loader does it where `Arch.transcodes` says (Q8_0
//! only where the fused kernels are affine-keyed), so those tensors run on MLX's own affine kernels; every other type
//! keeps its ggml blocks.
const std = @import("std");
const quants = @import("quants.zig");
const GgmlType = quants.GgmlType;

/// The code width an MLX affine layout holds `ty` at, null when `ty` is not an affine code per group.
pub fn bitsOf(ty: GgmlType) ?u32 {
    return switch (ty) {
        .q2_0 => 2,
        .q4_0, .q4_1, .q4_k => 4,
        .q5_0, .q5_1, .q5_k => 5,
        .q8_0 => 8,
        else => null,
    };
}

/// Weights per scale and bias of `ty`'s affine layout.
pub fn groupOf(ty: GgmlType) usize {
    return if (ty == .q2_0) 64 else 32;
}

fn half(b: []const u8, off: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[off..][0..2], .little))));
}

/// One row of whole GGUF blocks (`src`) as codes `q[in]` and one scale and bias per `groupOf(ty)` weights.
pub fn decodeRow(ty: GgmlType, src: []const u8, q: []u8, scale: []f32, bias: []f32) void {
    const be = ty.blockElems();
    const bb = ty.blockBytes();
    std.debug.assert(src.len == q.len / be * bb and scale.len == q.len / groupOf(ty) and bias.len == scale.len);
    for (0..q.len / be) |i| {
        const b = src[i * bb ..][0..bb];
        const qs = q[i * be ..][0..be];
        const g = i * be / groupOf(ty);
        switch (ty) {
            .q2_0 => {
                const d = half(b, 0);
                for (qs, 0..) |*o, j| o.* = (b[2 + j / 4] >> @intCast(j % 4 * 2)) & 3;
                scale[g] = d;
                bias[g] = -d;
            },
            .q4_0, .q4_1 => {
                const d = half(b, 0);
                const at: usize = if (ty == .q4_0) 2 else 4;
                for (b[at..][0..16], 0..) |v, j| {
                    qs[j] = v & 0xF;
                    qs[j + 16] = v >> 4;
                }
                scale[g] = d;
                bias[g] = if (ty == .q4_0) -8 * d else half(b, 2);
            },
            .q5_0, .q5_1 => {
                const d = half(b, 0);
                const at: usize = if (ty == .q5_0) 6 else 8;
                const qh = std.mem.readInt(u32, b[at - 4 ..][0..4], .little);
                for (b[at..][0..16], 0..) |v, j| {
                    const sh: u5 = @intCast(j);
                    qs[j] = (v & 0xF) | @as(u8, @truncate(((qh >> sh) << 4) & 0x10));
                    qs[j + 16] = (v >> 4) | @as(u8, @truncate((qh >> (sh + 12)) & 0x10));
                }
                scale[g] = d;
                bias[g] = if (ty == .q5_0) -16 * d else half(b, 2);
            },
            .q8_0 => {
                const d = half(b, 0);
                for (b[2..34], qs) |v, *o| o.* = v ^ 0x80; // int8 q -> q + 128
                scale[g] = d;
                bias[g] = -128 * d;
            },
            .q4_k, .q5_k => {
                const five = ty == .q5_k;
                const d = half(b, 0);
                const min = half(b, 2);
                const scales = b[4..16];
                const qh = b[16..48];
                const ql_all = if (five) b[48..176] else b[16..144];
                for (0..4) |j| {
                    const ql = ql_all[32 * j ..][0..32];
                    for (0..2) |h| {
                        const sub = 2 * j + h;
                        const sm = scaleMinK4(sub, scales);
                        const hbit = @as(u8, 1) << @intCast(sub);
                        for (ql, 0..) |v, l| {
                            var c: u8 = if (h == 0) v & 0xF else v >> 4;
                            if (five and qh[l] & hbit != 0) c += 16;
                            qs[32 * sub + l] = c;
                        }
                        scale[g + sub] = d * sm[0];
                        bias[g + sub] = -min * sm[1];
                    }
                }
            },
            else => unreachable,
        }
    }
}

fn scaleMinK4(j: usize, q: []const u8) struct { f32, f32 } {
    const f = struct {
        fn of(v: u8) f32 {
            return @floatFromInt(v);
        }
    }.of;
    if (j < 4) return .{ f(q[j] & 63), f(q[j + 4] & 63) };
    return .{ f((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)), f((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) };
}

/// Codes packed as `mx.quantize` packs them: code i at bit i * bits of a little-endian u32 stream.
pub fn packCodes(bits: u32, q: []const u8, out: []u32) void {
    std.debug.assert(q.len * bits == out.len * 32);
    @memset(out, 0);
    for (q, 0..) |c, i| {
        const at = i * bits;
        const w = at / 32;
        const sh: u5 = @intCast(at % 32);
        out[w] |= @as(u32, c) << sh;
        if (@as(usize, sh) + bits > 32) out[w + 1] |= @as(u32, c) >> @intCast(32 - @as(usize, sh));
    }
}

/// f32 to bf16, round to nearest even.
pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// Affine reconstruction of `fixture`'s blocks against ggml's own dequant (`quants.zig`'s fixtures).
fn expectAffineMatchesGgml(ty: GgmlType, comptime fixture: []const u8) !void {
    const data = @embedFile("fixtures/" ++ fixture);
    const n_blocks = data.len / (ty.blockBytes() + 4 * ty.blockElems());
    const n = n_blocks * ty.blockElems();
    const src = data[0 .. n_blocks * ty.blockBytes()];
    const want = data[src.len..];
    const a = std.testing.allocator;
    const q = try a.alloc(u8, n);
    defer a.free(q);
    const g = groupOf(ty);
    const sc = try a.alloc(f32, n / g);
    defer a.free(sc);
    const bi = try a.alloc(f32, n / g);
    defer a.free(bi);
    decodeRow(ty, src, q, sc, bi);
    const max = (@as(u32, 1) << @intCast(bitsOf(ty).?)) - 1;
    for (0..n) |i| {
        try std.testing.expect(q[i] <= max);
        const w: f32 = @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little));
        const got = sc[i / g] * @as(f32, @floatFromInt(q[i])) + bi[i / g];
        try std.testing.expectApproxEqAbs(w, got, 1e-5 * @max(1.0, @abs(w)));
    }
}

test "every affine-coded GGUF type decodes to ggml's own weights" {
    try expectAffineMatchesGgml(.q4_0, "q4_0.bin");
    try expectAffineMatchesGgml(.q4_1, "q4_1.bin");
    try expectAffineMatchesGgml(.q5_0, "q5_0.bin");
    try expectAffineMatchesGgml(.q5_1, "q5_1.bin");
    try expectAffineMatchesGgml(.q8_0, "q8_0.bin");
    try expectAffineMatchesGgml(.q4_k, "q4_k.bin");
    try expectAffineMatchesGgml(.q5_k, "q5_k.bin");
    try expectAffineMatchesGgml(.q2_0, "q2_0.bin");
    try std.testing.expect(bitsOf(.q6_k) == null and bitsOf(.iq4_nl) == null);
    try std.testing.expect(groupOf(.q2_0) == 64 and groupOf(.q4_k) == 32);
}

test "packCodes packs densely, straddling words at 5 bits" {
    var q: [32]u8 = undefined;
    for (&q, 0..) |*c, i| c.* = @intCast((i * 7 + 3) % 32);
    var out: [5]u32 = undefined;
    packCodes(5, &q, &out);
    for (q, 0..) |c, i| {
        const bit = i * 5;
        const lo: u64 = out[bit / 32];
        const hi: u64 = if (bit / 32 + 1 < out.len) out[bit / 32 + 1] else 0;
        try std.testing.expectEqual(@as(u64, c), ((lo | hi << 32) >> @intCast(bit % 32)) & 31);
    }
}
