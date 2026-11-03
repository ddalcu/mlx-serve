//! ggml tensor types we serve, their block geometry, and the CPU reference
//! dequant. The Metal kernels are checked against this, and this is checked
//! against ggml itself (src/fixtures, see scripts/gen_fixtures.sh).
const std = @import("std");
const tables = @import("iq_tables.zig");

/// Values are the on-disk GGUF tensor type ids.
pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    iq1_m = 29,
    bf16 = 30,
    mxfp4 = 39,
    _,

    pub fn supported(self: GgmlType) bool {
        return std.enums.tagName(GgmlType, self) != null;
    }

    /// Weights per block. Unsupported types return 0.
    pub fn blockElems(self: GgmlType) usize {
        return switch (self) {
            .f32, .f16, .bf16 => 1,
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .iq4_nl, .mxfp4 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq3_s, .iq2_s, .iq4_xs, .iq1_s, .iq1_m => 256,
            _ => 0,
        };
    }

    /// Bytes per block. Unsupported types return 0.
    pub fn blockBytes(self: GgmlType) usize {
        return switch (self) {
            .f32 => 4,
            .f16, .bf16 => 2,
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .iq2_xxs => 66,
            .iq3_xxs => 98,
            .iq4_nl => 18,
            .iq3_s => 110,
            .iq2_s => 82,
            .iq2_xs => 74,
            .iq4_xs => 136,
            .mxfp4 => 17,
            .iq1_s => 50,
            .iq1_m => 56,
            _ => 0,
        };
    }

    /// Weights per decode unit of the Metal kernels, see metal/blocks.h.
    pub fn unitElems(self: GgmlType) usize {
        return switch (self) {
            .q4_k, .q5_k => 64,
            .q2_k, .q3_k, .q6_k => 128,
            else => @min(self.blockElems(), 32),
        };
    }

    pub fn isQuantized(self: GgmlType) bool {
        return self.blockElems() > 1;
    }
};

pub const Error = error{ UnsupportedType, BadLength };

/// Dequantize whole blocks of a quantized type into f32.
pub fn dequantize(ty: GgmlType, src: []const u8, dst: []f32) Error!void {
    if (!ty.isQuantized()) return error.UnsupportedType;
    const be = ty.blockElems();
    const bb = ty.blockBytes();
    if (dst.len % be != 0 or src.len != dst.len / be * bb) return error.BadLength;
    var i: usize = 0;
    while (i < dst.len / be) : (i += 1) {
        const blk = src[i * bb ..][0..bb];
        const y = dst[i * be ..][0..be];
        switch (ty) {
            .q4_0 => q4_0(blk, y, -8, 0, 2),
            .q4_1 => q4_0(blk, y, 0, half(blk, 2), 4),
            .q5_0 => q5_0(blk, y, -16, 0, 6),
            .q5_1 => q5_0(blk, y, 0, half(blk, 2), 8),
            .q8_0 => q8_0(blk, y),
            .q2_k => q2K(blk, y),
            .q3_k => q3K(blk, y),
            .q4_k => q45K(blk, y, false),
            .q5_k => q45K(blk, y, true),
            .q6_k => q6K(blk, y),
            .iq2_xxs => iq2Xxs(blk, y),
            .iq2_s => iq2S(blk, y),
            .iq3_xxs => iq3Xxs(blk, y),
            .iq3_s => iq3S(blk, y),
            .iq4_nl => iq4Nibbles(blk[2..18], y, half(blk, 0)),
            .iq4_xs => iq4Xs(blk, y),
            .iq2_xs => iq2Xs(blk, y),
            .iq1_s => iq1S(blk, y),
            .iq1_m => iq1M(blk, y),
            .mxfp4 => mxfp4(blk, y),
            else => unreachable,
        }
    }
}

fn half(b: []const u8, off: usize) f32 {
    return @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[off..][0..2], .little))));
}

fn int(v: anytype) f32 {
    return @floatFromInt(v);
}

/// The grids pack one magnitude per byte, low byte first.
fn gridByte(entry: anytype, j: usize) f32 {
    return int(@as(u8, @truncate(entry >> @intCast(8 * j))));
}

fn sign(bits: u8, j: usize) f32 {
    return if (bits & tables.kmask_iq2xs[j] != 0) -1.0 else 1.0;
}

/// Q4_0 / Q4_1: 16 nibble bytes at `qs`, low nibbles first; y = (q + off) * d + m.
fn q4_0(b: []const u8, y: []f32, off: f32, m: f32, qs: usize) void {
    const d = half(b, 0);
    for (b[qs..][0..16], 0..) |q, j| {
        y[j] = (int(q & 0xF) + off) * d + m;
        y[j + 16] = (int(q >> 4) + off) * d + m;
    }
}

/// Q5_0 / Q5_1: fifth bits in the u32 before the nibbles, bit j for weight j, bit j + 16 for weight j + 16.
fn q5_0(b: []const u8, y: []f32, off: f32, m: f32, qs: usize) void {
    const d = half(b, 0);
    const qh = std.mem.readInt(u32, b[qs - 4 ..][0..4], .little);
    for (b[qs..][0..16], 0..) |q, j| {
        const sh: u5 = @intCast(j);
        y[j] = (int((q & 0xF) | @as(u8, @truncate(((qh >> sh) << 4) & 0x10))) + off) * d + m;
        y[j + 16] = (int((q >> 4) | @as(u8, @truncate((qh >> (sh + 12)) & 0x10))) + off) * d + m;
    }
}

const kvalues_fp4 = [16]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

/// One E8M0 scale byte (ggml scales it by half) and 16 bytes of FP4 nibbles, low nibbles first.
fn mxfp4(b: []const u8, y: []f32) void {
    const e = b[0];
    const bits: u32 = if (e < 2) @as(u32, 0x00200000) << @intCast(e) else @as(u32, e - 1) << 23;
    const d: f32 = @bitCast(bits);
    for (b[1..17], 0..) |q, j| {
        y[j] = int(kvalues_fp4[q & 0xF]) * d;
        y[j + 16] = int(kvalues_fp4[q >> 4]) * d;
    }
}

fn q8_0(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    for (y, b[2..34]) |*o, q| o.* = int(@as(i8, @bitCast(q))) * d;
}

fn q2K(b: []const u8, y: []f32) void {
    const scales = b[0..16];
    const d = half(b, 80);
    const min = half(b, 82);
    var yi: usize = 0;
    for (0..2) |n| {
        const q = b[16 + 32 * n ..][0..32];
        for (0..4) |j| {
            for (0..2) |h| {
                const sc = scales[n * 8 + j * 2 + h];
                const dl = d * int(sc & 0xF);
                const ml = min * int(sc >> 4);
                for (q[16 * h ..][0..16]) |v| {
                    y[yi] = dl * int((v >> @intCast(2 * j)) & 3) - ml;
                    yi += 1;
                }
            }
        }
    }
}

fn q3K(b: []const u8, y: []f32) void {
    const hm = b[0..32];
    const d_all = half(b, 108);
    const kmask1: u32 = 0x03030303;
    const kmask2: u32 = 0x0f0f0f0f;
    var aux: [4]u32 = undefined;
    for (0..3) |i| aux[i] = std.mem.readInt(u32, b[96 + 4 * i ..][0..4], .little);
    const tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    var yi: usize = 0;
    for (0..2) |n| {
        const q = b[32 + 32 * n ..][0..32];
        for (0..4) |j| {
            const m = @as(u8, 1) << @intCast(n * 4 + j);
            for (0..2) |h| {
                const is = n * 8 + j * 2 + h;
                const sc: i8 = @bitCast(@as(u8, @truncate(aux[is / 4] >> @intCast(8 * (is % 4)))));
                const dl = d_all * int(@as(i32, sc) - 32);
                for (0..16) |l| {
                    const low: i32 = (q[16 * h + l] >> @intCast(2 * j)) & 3;
                    const high: i32 = if (hm[16 * h + l] & m != 0) 0 else 4;
                    y[yi] = dl * int(low - high);
                    yi += 1;
                }
            }
        }
    }
}

fn scaleMinK4(j: usize, q: []const u8) struct { f32, f32 } {
    if (j < 4) return .{ int(q[j] & 63), int(q[j + 4] & 63) };
    return .{
        int((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)),
        int((q[j + 4] >> 4) | ((q[j] >> 6) << 4)),
    };
}

/// Q4_K and Q5_K share the layout, Q5_K adds a high-bit plane (qh).
fn q45K(b: []const u8, y: []f32, comptime five: bool) void {
    const d = half(b, 0);
    const min = half(b, 2);
    const scales = b[4..16];
    const qh = b[16..48];
    const qs = if (five) b[48..176] else b[16..144];
    var yi: usize = 0;
    for (0..4) |j| {
        const ql = qs[32 * j ..][0..32];
        for (0..2) |h| {
            const sm = scaleMinK4(2 * j + h, scales);
            const dl = d * sm[0];
            const ml = min * sm[1];
            const hbit = @as(u8, 1) << @intCast(2 * j + h);
            for (ql, 0..) |v, l| {
                var q: u8 = if (h == 0) v & 0xF else v >> 4;
                if (five and qh[l] & hbit != 0) q += 16;
                y[yi] = dl * int(q) - ml;
                yi += 1;
            }
        }
    }
}

fn q6K(b: []const u8, y: []f32) void {
    const d = half(b, 208);
    for (0..2) |n| {
        const ql = b[64 * n ..][0..64];
        const qh = b[128 + 32 * n ..][0..32];
        const sc = b[192 + 8 * n ..][0..8];
        for (0..32) |l| {
            const is = l / 16;
            const lows = [4]u8{ ql[l] & 0xF, ql[l + 32] & 0xF, ql[l] >> 4, ql[l + 32] >> 4 };
            for (lows, 0..) |low, k| {
                const q: i32 = @as(i32, low | (((qh[l] >> @intCast(2 * k)) & 3) << 4)) - 32;
                const s: i8 = @bitCast(sc[is + 2 * k]);
                y[128 * n + 32 * k + l] = d * int(s) * int(q);
            }
        }
    }
}

fn iq2Xxs(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    for (0..8) |ib32| {
        const aux8 = b[2 + 8 * ib32 ..][0..8];
        const aux1 = std.mem.readInt(u32, aux8[4..8], .little);
        const db = d * (0.5 + int(aux1 >> 28)) * 0.25;
        for (0..4) |l| {
            const grid = tables.iq2xxs_grid[aux8[l]];
            const signs = tables.ksigns_iq2xs[(aux1 >> @intCast(7 * l)) & 127];
            for (0..8) |j| y[32 * ib32 + 8 * l + j] = db * gridByte(grid, j) * sign(signs, j);
        }
    }
}

fn iq2S(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    const qs = b[2..34];
    const signs = b[34..66];
    const qh = b[66..74];
    const scales = b[74..82];
    for (0..8) |ib32| {
        const db = [2]f32{
            d * (0.5 + int(scales[ib32] & 0xF)) * 0.25,
            d * (0.5 + int(scales[ib32] >> 4)) * 0.25,
        };
        for (0..4) |l| {
            const idx = @as(usize, qs[4 * ib32 + l]) | ((@as(usize, qh[ib32]) << @intCast(8 - 2 * l)) & 0x300);
            const grid = tables.iq2s_grid[idx];
            for (0..8) |j| y[32 * ib32 + 8 * l + j] = db[l / 2] * gridByte(grid, j) * sign(signs[4 * ib32 + l], j);
        }
    }
}

fn iq3Xxs(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    const qs = b[2..66];
    for (0..8) |ib32| {
        const aux = std.mem.readInt(u32, b[66 + 4 * ib32 ..][0..4], .little);
        const db = d * (0.5 + int(aux >> 28)) * 0.5;
        for (0..4) |l| {
            const signs = tables.ksigns_iq2xs[(aux >> @intCast(7 * l)) & 127];
            for (0..2) |g| {
                const grid = tables.iq3xxs_grid[qs[8 * ib32 + 2 * l + g]];
                for (0..4) |j| y[32 * ib32 + 8 * l + 4 * g + j] = db * gridByte(grid, j) * sign(signs, 4 * g + j);
            }
        }
    }
}

fn iq3S(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    const qs = b[2..66];
    const qh = b[66..74];
    const signs = b[74..106];
    const scales = b[106..110];
    for (0..8) |ib32| {
        const nib = if (ib32 % 2 == 0) scales[ib32 / 2] & 0xF else scales[ib32 / 2] >> 4;
        const db = d * int(1 + 2 * @as(u32, nib));
        for (0..4) |l| {
            for (0..2) |g| {
                const hi = (@as(usize, qh[ib32]) << @intCast(8 - g - 2 * l)) & 256;
                const grid = tables.iq3s_grid[@as(usize, qs[8 * ib32 + 2 * l + g]) | hi];
                for (0..4) |j| y[32 * ib32 + 8 * l + 4 * g + j] = db * gridByte(grid, j) * sign(signs[4 * ib32 + l], 4 * g + j);
            }
        }
    }
}

/// 32 weights from 16 bytes of codebook nibbles, shared by IQ4_NL and IQ4_XS.
fn iq4Nibbles(qs: *const [16]u8, y: []f32, d: f32) void {
    for (qs, 0..) |q, j| {
        y[j] = d * int(tables.kvalues_iq4nl[q & 0xF]);
        y[j + 16] = d * int(tables.kvalues_iq4nl[q >> 4]);
    }
}

fn iq4Xs(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    const scales_h = std.mem.readInt(u16, b[2..4], .little);
    for (0..8) |ib| {
        const lo = (b[4 + ib / 2] >> @intCast(4 * (ib % 2))) & 0xF;
        const ls: i32 = @as(i32, lo) | (@as(i32, (scales_h >> @intCast(2 * ib)) & 3) << 4);
        iq4Nibbles(b[8 + 16 * ib ..][0..16], y[32 * ib ..][0..32], d * int(ls - 32));
    }
}

fn iq2Xs(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    const scales = b[66..74];
    for (0..8) |ib32| {
        const db = [2]f32{
            d * (0.5 + int(scales[ib32] & 0xF)) * 0.25,
            d * (0.5 + int(scales[ib32] >> 4)) * 0.25,
        };
        for (0..4) |l| {
            const q = std.mem.readInt(u16, b[2 + 2 * (4 * ib32 + l) ..][0..2], .little);
            const grid = tables.iq2xs_grid[q & 511];
            const signs = tables.ksigns_iq2xs[q >> 9];
            for (0..8) |j| y[32 * ib32 + 8 * l + j] = db[l / 2] * gridByte(grid, j) * sign(signs, j);
        }
    }
}

const iq1_delta = 0.125;

/// 8 weights of the IQ1 grid: byte j of the entry holds weight j + 1 in its low
/// nibble and weight j + 4 + 1 in its high one.
fn iq1Group(y: []f32, idx: usize, dl: f32, negative: bool) void {
    const grid = tables.iq1s_grid_gpu[idx];
    const delta: f32 = if (negative) -iq1_delta else iq1_delta;
    for (0..4) |j| {
        const byte: u8 = @truncate(grid >> @intCast(8 * j));
        y[j] = dl * (int(byte & 0xF) - 1.0 + delta);
        y[j + 4] = dl * (int(byte >> 4) - 1.0 + delta);
    }
}

fn iq1S(b: []const u8, y: []f32) void {
    const d = half(b, 0);
    for (0..8) |ib| {
        const qh = std.mem.readInt(u16, b[34 + 2 * ib ..][0..2], .little);
        const dl = d * int(2 * ((qh >> 12) & 7) + 1);
        for (0..4) |l| {
            const idx = @as(usize, b[2 + 4 * ib + l]) | (@as(usize, (qh >> @intCast(3 * l)) & 7) << 8);
            iq1Group(y[32 * ib + 8 * l ..], idx, dl, qh & 0x8000 != 0);
        }
    }
}

fn iq1M(b: []const u8, y: []f32) void {
    var sc: [4]u16 = undefined;
    for (&sc, 0..) |*v, i| v.* = std.mem.readInt(u16, b[48 + 2 * i ..][0..2], .little);
    // The block's f16 scale is the top nibble of each scale word.
    const d_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
    for (0..8) |ib| {
        for (0..4) |l| {
            const dl = d * int(2 * ((sc[ib / 2] >> @intCast(6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
            const qh = b[32 + 2 * ib + l / 2] >> @intCast(4 * (l % 2));
            const idx = @as(usize, b[4 * ib + l]) | (@as(usize, qh & 7) << 8);
            iq1Group(y[32 * ib + 8 * l ..], idx, dl, qh & 8 != 0);
        }
    }
}

/// Fixture = N raw blocks followed by ggml's f32 dequant of them.
fn expectMatchesGgml(ty: GgmlType, comptime fixture: []const u8) !void {
    const data = @embedFile("fixtures/" ++ fixture);
    const n_blocks = data.len / (ty.blockBytes() + 4 * ty.blockElems());
    const n = n_blocks * ty.blockElems();
    const src = data[0 .. n_blocks * ty.blockBytes()];
    const want = data[src.len..];

    const got = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(got);
    try dequantize(ty, src, got);
    for (got, 0..) |g, i| {
        const w: f32 = @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little));
        // ggml's C build may fuse a*b-c, so allow last-bit drift.
        try std.testing.expectApproxEqRel(w, g, 1e-5);
    }
}

test "dequant matches ggml for every supported type" {
    try expectMatchesGgml(.q4_0, "q4_0.bin");
    try expectMatchesGgml(.q4_1, "q4_1.bin");
    try expectMatchesGgml(.q5_0, "q5_0.bin");
    try expectMatchesGgml(.q5_1, "q5_1.bin");
    try expectMatchesGgml(.q8_0, "q8_0.bin");
    try expectMatchesGgml(.q2_k, "q2_k.bin");
    try expectMatchesGgml(.q3_k, "q3_k.bin");
    try expectMatchesGgml(.q4_k, "q4_k.bin");
    try expectMatchesGgml(.q5_k, "q5_k.bin");
    try expectMatchesGgml(.q6_k, "q6_k.bin");
    try expectMatchesGgml(.iq2_xxs, "iq2_xxs.bin");
    try expectMatchesGgml(.iq2_s, "iq2_s.bin");
    try expectMatchesGgml(.iq3_xxs, "iq3_xxs.bin");
    try expectMatchesGgml(.iq3_s, "iq3_s.bin");
    try expectMatchesGgml(.iq4_nl, "iq4_nl.bin");
    try expectMatchesGgml(.iq4_xs, "iq4_xs.bin");
    try expectMatchesGgml(.iq2_xs, "iq2_xs.bin");
    try expectMatchesGgml(.iq1_s, "iq1_s.bin");
    try expectMatchesGgml(.iq1_m, "iq1_m.bin");
    try expectMatchesGgml(.mxfp4, "mxfp4.bin");
}

test "dequantize rejects a short buffer and unknown types" {
    var y: [32]f32 = undefined;
    try std.testing.expectError(error.BadLength, dequantize(.q8_0, &@as([33]u8, @splat(0)), &y));
    try std.testing.expectError(error.UnsupportedType, dequantize(@fromBackingInt(@intCast(15)), &.{}, &y));
    try std.testing.expect(!@as(GgmlType, @fromBackingInt(@intCast(15))).supported());
}
