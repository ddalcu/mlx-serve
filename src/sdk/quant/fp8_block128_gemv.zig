//! `fp8_block128`'s MLX-free facts: the config entry that names it, and the decode GEMV's geometry and its arithmetic
//! on the host: the plan a projection's
//! `kernels/linear/fp8_block128_gemv.metal` launch runs, and the kernel's summation order emulated for its parity.
//! Every output's sum is fixed by the plan's K partitions (KS) and the text, never by the call's rows (M), R or RG, so
//! row m of an M-row call is the M = 1 call's row bit for bit.

const std = @import("std");

/// The release's scale grid: one f32 per 64 output rows x 128 input columns.
pub const scale_rows: u32 = 64;
pub const scale_cols: u32 = 128;
/// A projection's `quantization` entry naming e4m3 on the 64 x 128 scale grid.
pub fn isEntry(o: std.json.ObjectMap) bool {
    const mode = o.get("mode") orelse return false;
    const bits = o.get("bits") orelse return false;
    const blk = o.get("block") orelse return false;
    if (mode != .string or !std.mem.eql(u8, mode.string, "fp8_e4m3") or bits != .integer or bits.integer != 8) return false;
    if (blk != .array or blk.array.items.len != 2) return false;
    const r = blk.array.items[0];
    const c = blk.array.items[1];
    return r == .integer and r.integer == scale_rows and c == .integer and c.integer == scale_cols;
}

/// The widest call the GEMV takes (decode, verify windows, a head's rows); wider calls take the prompt route.
pub const max_rows: c_int = 8;
/// Codes per lane per step and the step (32 lanes): a 256-column step, two scale blocks; a lane's 8 codes lie in one
/// 128-column block.
pub const kv: u32 = 8;
pub const step: u32 = 32 * kv;
/// The simdgroups that fill the GPU once: a site with fewer row groups splits K until it has them.
pub const target_simdgroups: u32 = 1024;

/// R rows per simdgroup (within one 64-row scale row), KS K partitions per row group (summed in order through
/// threadgroup memory), RG row groups per threadgroup.
pub const Gemv = struct { r: c_int = 4, ks: c_int = 1, rg: c_int = 4 };

/// The site plan of an [n, k] projection, or null when the text does not cover the shape.
pub fn plan(n: u32, k: u32) ?Gemv {
    const r: u32 = 4;
    if (n % scale_rows != 0 or k % scale_cols != 0 or k % step != 0 or n % r != 0) return null;
    var ks: u32 = 1;
    while (ks < 8 and (n / r) * ks < target_simdgroups and (k / step) % (ks * 2) == 0) ks *= 2;
    var rg: u32 = @max(1, 4 / ks);
    while (rg > 1 and n % (r * rg) != 0) rg /= 2;
    return .{ .r = @intCast(r), .ks = @intCast(ks), .rg = @intCast(rg) };
}

/// The plan a call of `m` rows runs: the site's, with R alone changed where it measured faster (2 at the 4-row verify;
/// 1 on single-partition K = 4096 sites at 5 and 6 rows). KS and RG stay the site's, so the sums do too.
pub fn planAt(site: Gemv, n: u32, k: u32, m: u32) Gemv {
    const r: c_int = switch (m) {
        4 => 2,
        5, 6 => if (site.ks == 1 and k == 4096) 1 else site.r,
        else => site.r,
    };
    var p = site;
    p.r = r;
    if (n % @as(u32, @intCast(p.r * p.rg)) != 0 or @rem(64, p.r) != 0) return site;
    return p;
}

pub const Launch = struct { grid: [3]c_int, tg: [3]c_int };

/// Threadgroups of RG x KS simdgroups; grid y = one threadgroup per R * RG rows.
pub fn launch(p: Gemv, n: c_int) Launch {
    const groups = @divExact(n, p.r * p.rg);
    return .{ .grid = .{ 32, groups * p.rg * p.ks, 1 }, .tg = .{ 32, p.rg * p.ks, 1 } };
}

/// The kernels' e4m3 decode: (code & 127) << 7 with the sign at bit 15, read as a half (the code's value / 256).
pub fn decodeHalf(code: u8) f32 {
    const bits: u16 = (@as(u16, code & 0x7F) << 7) | (@as(u16, code & 0x80) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// The GEMV's weight: f32 code x scale, one rounding, as the kernel forms it.
pub fn weight32(code: u8, scale: f32) f32 {
    return decodeHalf(code) * (scale * 256);
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// Round to nearest even (no NaN reaches here).
pub fn f32ToBf16(v: f32) u16 {
    const r: u32 = @bitCast(v);
    return @truncate((r +% 0x7fff +% ((r >> 16) & 1)) >> 16);
}

/// One output of the GEMV in the kernel's order over weight row `w` (`weight32`) and BF16 input row `xr`: per lane, fma
/// over steps then codes; the xor butterfly; the K partitions in order. The f32 sum the kernel rounds to BF16 once.
pub fn hostDot(p: Gemv, xr: []const u16, w: []const f32) f32 {
    const ks: usize = @intCast(p.ks);
    const kp = xr.len / ks;
    var total: f32 = 0;
    for (0..ks) |part| {
        var lanes: [32]f32 = undefined;
        for (0..32) |l| {
            var acc: f32 = 0;
            const k0 = part * kp + kv * l;
            var st: usize = 0;
            while (st < kp / step) : (st += 1) {
                const kk = k0 + st * step;
                for (0..kv) |i| acc = @mulAdd(f32, bf16ToF32(xr[kk + i]), w[kk + i], acc);
            }
            lanes[l] = acc;
        }
        for ([_]usize{ 16, 8, 4, 2, 1 }) |off| {
            var next: [32]f32 = undefined;
            for (0..32) |l| next[l] = lanes[l] + lanes[l ^ off];
            lanes = next;
        }
        total = if (part == 0) lanes[0] else total + lanes[0];
    }
    return total;
}

// ── tests ──

const testing = std.testing;

test "fp8_block128 gemv: the site plans of MiMo's FP8 projections, and the per-width R that keeps their KS and RG" {
    const Site = struct { n: u32, k: u32, want: Gemv };
    const sites = [_]Site{
        .{ .n = 12288, .k = 4096, .want = .{ .r = 4, .ks = 1, .rg = 4 } },
        .{ .n = 768, .k = 4096, .want = .{ .r = 4, .ks = 8, .rg = 1 } },
        .{ .n = 512, .k = 4096, .want = .{ .r = 4, .ks = 8, .rg = 1 } },
        .{ .n = 1536, .k = 4096, .want = .{ .r = 4, .ks = 4, .rg = 1 } },
        .{ .n = 1024, .k = 4096, .want = .{ .r = 4, .ks = 4, .rg = 1 } },
        .{ .n = 16384, .k = 4096, .want = .{ .r = 4, .ks = 1, .rg = 4 } },
        .{ .n = 4096, .k = 16384, .want = .{ .r = 4, .ks = 1, .rg = 4 } },
    };
    for (sites) |s| {
        const p = plan(s.n, s.k).?;
        try testing.expectEqual(s.want, p);
        for (1..max_rows + 1) |m| {
            const q = planAt(p, s.n, s.k, @intCast(m));
            try testing.expect(q.ks == p.ks and q.rg == p.rg and @rem(64, q.r) == 0 and s.n % @as(u32, @intCast(q.r * q.rg)) == 0);
        }
    }
    try testing.expectEqual(@as(c_int, 2), planAt(plan(768, 4096).?, 768, 4096, 4).r);
    try testing.expectEqual(@as(c_int, 1), planAt(plan(12288, 4096).?, 12288, 4096, 5).r);
    try testing.expect(plan(12288, 4000) == null and plan(12290, 4096) == null and plan(96, 4096) == null);
}
