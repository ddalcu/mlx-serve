//! Cheap per-step video previews for SSE progress events.
//!
//! A full VAE decode every denoise step is not safe (H3 stages the text
//! encoder and DiT because they cannot both stay resident). This is the
//! no-extra-weights fallback from issue #208: the PUBLISHED linear
//! latent-to-RGB projection for the backend's latent space (`src/latent_rgb.zig`,
//! generated from ComfyUI's fitted tables), optional bilinear resize, then
//! JPEG. Hermetic — no MLX.
//!
//! The projection is the reference's, arithmetic included: `rgb = W·latent + b`
//! then `(rgb + 1) / 2` clamped and scaled to 0–255. There is NO per-frame
//! contrast stretch — the fit already calibrates the output range, and stretching
//! on top both undoes the fit and makes brightness jump between steps.
//!
//! Two tiers of proof, because the fixture oracle below can only show we
//! reproduce ComfyUI and would be just as green if ComfyUI's table were a hue
//! wheel: the PERCEPTUAL bar (`boxDownsampleRgb` + `rgbCorrelation` +
//! `goldenAngleControlMap`, driven by the live tests in `minimax_h3_vae.zig`
//! and `ltx_video.zig`) correlates the projection against a real VAE decode.

const std = @import("std");
const jpeg = @import("jpeg.zig");
const latent_rgb = @import("latent_rgb.zig");

pub const Map = latent_rgb.Map;
/// LTX-2 (2.3 and 2.5), 128 channels. NOT LTX-0.9's `LTXV` table — same
/// geometry, different latent space (see src/latent_rgb.zig).
pub const ltx_av = latent_rgb.ltx_av;
/// MiniMax-H3, 24 channels.
pub const minimax_h3 = latent_rgb.minimax_h3;

/// Opt-in request knobs. Default behaviour is off.
pub const Opts = struct {
    enabled: bool = false,
    /// How many temporal slices to show. 1 = one JPEG (mid clip, or frame 0
    /// for I2V). >1 = evenly spaced slices packed as a horizontal JPEG
    /// filmstrip. Animated WebP is a later cut.
    frames: u32 = 1,
    /// Max width or height in px. 0 = latent native size.
    max_side: u32 = 256,

    pub const max_frames: u32 = 8;
    pub const max_side_cap: u32 = 1024;

    pub fn normalize(self: Opts) Opts {
        return .{
            .enabled = self.enabled,
            .frames = if (self.frames == 0) 1 else @min(self.frames, max_frames),
            .max_side = @min(self.max_side, max_side_cap),
        };
    }
};

/// One encoded preview image. Caller owns `jpeg`.
pub const Encoded = struct {
    jpeg: []u8,
    w: u32,
    h: u32,
    mime: []const u8 = "image/jpeg",
};

/// Temporal indices into a clip of length `t`. `first_frame` pins the first
/// slice at 0 (I2V / keyframe). Otherwise a single slice is the midpoint.
pub fn temporalIndices(out: []u32, t: u32, n_want: u32, first_frame: bool) []u32 {
    if (t == 0 or out.len == 0) return out[0..0];
    const n: u32 = @min(n_want, @min(t, @as(u32, @intCast(out.len))));
    if (n == 1) {
        out[0] = if (first_frame) 0 else (t - 1) / 2;
        return out[0..1];
    }
    const denom = n - 1;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        out[i] = (i * (t - 1)) / denom;
    }
    return out[0..n];
}

/// BCFHW latent (B=1) → JPEG. `latent` is C*T*H*W f32, channel-major, where C
/// is `map.channels()`. `first_frame` selects temporal index 0 when
/// `opts.frames == 1`.
///
/// Callers that can slice on the GPU should pass only the frames they want
/// (`t` = the slice count, `frames` = the same number): copying a whole
/// 128-channel volume to the host every step is the expensive part, not this.
pub fn jpegFromLatent(
    allocator: std.mem.Allocator,
    map: Map,
    latent: []const f32,
    t: u32,
    h: u32,
    w: u32,
    opts: Opts,
    first_frame: bool,
) !Encoded {
    const o = opts.normalize();
    const c = map.channels();
    const need = @as(usize, c) * @as(usize, t) * @as(usize, h) * @as(usize, w);
    if (latent.len < need or c == 0 or t == 0 or h == 0 or w == 0) return error.BadLatentShape;

    var idx_buf: [Opts.max_frames]u32 = undefined;
    const idx = temporalIndices(&idx_buf, t, o.frames, first_frame);

    const rgb_native = try allocator.alloc(u8, @as(usize, idx.len) * @as(usize, h) * @as(usize, w) * 3);
    defer allocator.free(rgb_native);
    for (idx, 0..) |ti, fi| {
        latentSliceToRgb(map, latent, t, h, w, ti, rgb_native[fi * @as(usize, h) * @as(usize, w) * 3 ..][0 .. @as(usize, h) * @as(usize, w) * 3]);
    }

    const strip_w = w * @as(u32, @intCast(idx.len));
    const strip_h = h;
    const resized = try resizeMaxSide(allocator, rgb_native, strip_w, strip_h, o.max_side);
    defer if (resized.owned) allocator.free(resized.rgb);

    const jpg = try jpeg.encodeRgb(allocator, resized.rgb, resized.w, resized.h, 72);
    return .{ .jpeg = jpg, .w = resized.w, .h = resized.h };
}

/// SSE `data:` payload (no trailing blank line). Stage is JSON-escaped.
pub fn formatProgressJson(
    allocator: std.mem.Allocator,
    stage: []const u8,
    step: u32,
    total: u32,
    preview: ?Encoded,
) ![]u8 {
    var esc_buf: [256]u8 = undefined;
    const esc = jsonEscape(&esc_buf, stage);
    if (preview) |p| {
        const b64_len = std.base64.standard.Encoder.calcSize(p.jpeg.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, p.jpeg);
        return std.fmt.allocPrint(
            allocator,
            "{{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d},\"preview\":\"{s}\",\"mime\":\"{s}\",\"w\":{d},\"h\":{d}}}",
            .{ esc, step, total, b64, p.mime, p.w, p.h },
        );
    }
    return std.fmt.allocPrint(
        allocator,
        "{{\"type\":\"progress\",\"stage\":\"{s}\",\"step\":{d},\"total\":{d}}}",
        .{ esc, step, total },
    );
}

fn jsonEscape(out: []u8, msg: []const u8) []const u8 {
    var n: usize = 0;
    for (msg) |ch| {
        var one: [1]u8 = .{if (ch < 0x20) ' ' else ch};
        const esc: []const u8 = switch (ch) {
            '"' => "\\\"",
            '\\' => "\\\\",
            '\n' => "\\n",
            '\r' => "\\r",
            '\t' => "\\t",
            else => one[0..1],
        };
        if (n + esc.len > out.len) break;
        @memcpy(out[n..][0..esc.len], esc);
        n += esc.len;
    }
    return out[0..n];
}

/// One temporal slice of a `[C,T,H,W]` latent through the map, into RGB8.
/// Byte-for-byte the reference's `Latent2RGBPreviewer` + `preview_to_image`.
pub fn latentSliceToRgb(map: Map, latent: []const f32, t: u32, h: u32, w: u32, ti: u32, rgb: []u8) void {
    const hw: usize = @as(usize, h) * @as(usize, w);
    const thw: usize = @as(usize, t) * hw;
    var pix: usize = 0;
    while (pix < hw) : (pix += 1) {
        var acc = map.bias;
        for (map.factors, 0..) |f, ci| {
            const v = latent[ci * thw + @as(usize, ti) * hw + pix];
            acc[0] += v * f[0];
            acc[1] += v * f[1];
            acc[2] += v * f[2];
        }
        rgb[pix * 3 + 0] = toU8(acc[0]);
        rgb[pix * 3 + 1] = toU8(acc[1]);
        rgb[pix * 3 + 2] = toU8(acc[2]);
    }
}

/// `((x + 1) / 2).clamp(0, 1) * 255` then TRUNCATE — torch's `.to(uint8)`
/// truncates, and rounding here would disagree with the reference at every
/// value that lands on an integer boundary.
fn toU8(x: f32) u8 {
    const y = (x + 1.0) * 0.5;
    const z = @min(@max(y, 0.0), 1.0);
    return @intFromFloat(z * 255.0);
}

const Resize = struct {
    rgb: []const u8,
    w: u32,
    h: u32,
    owned: bool,
};

fn resizeMaxSide(allocator: std.mem.Allocator, rgb: []const u8, w: u32, h: u32, max_side: u32) !Resize {
    if (max_side == 0) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const long: u32 = @max(w, h);
    if (long <= max_side) return .{ .rgb = rgb, .w = w, .h = h, .owned = false };
    const nw: u32 = @max(1, (w * max_side) / long);
    const nh: u32 = @max(1, (h * max_side) / long);
    const out = try allocator.alloc(u8, @as(usize, nw) * @as(usize, nh) * 3);
    bilinear(rgb, w, h, out, nw, nh);
    return .{ .rgb = out, .w = nw, .h = nh, .owned = true };
}

fn bilinear(src: []const u8, sw: u32, sh: u32, dst: []u8, dw: u32, dh: u32) void {
    const x_scale = @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw));
    const y_scale = @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh));
    var y: u32 = 0;
    while (y < dh) : (y += 1) {
        const fy = (@as(f32, @floatFromInt(y)) + 0.5) * y_scale - 0.5;
        const y0 = @as(u32, @intFromFloat(@max(fy, 0)));
        const y1 = @min(y0 + 1, sh - 1);
        const ty = fy - @as(f32, @floatFromInt(y0));
        var x: u32 = 0;
        while (x < dw) : (x += 1) {
            const fx = (@as(f32, @floatFromInt(x)) + 0.5) * x_scale - 0.5;
            const x0 = @as(u32, @intFromFloat(@max(fx, 0)));
            const x1 = @min(x0 + 1, sw - 1);
            const tx = fx - @as(f32, @floatFromInt(x0));
            var ch: u32 = 0;
            while (ch < 3) : (ch += 1) {
                const a = samp(src, sw, x0, y0, ch);
                const b = samp(src, sw, x1, y0, ch);
                const c = samp(src, sw, x0, y1, ch);
                const d = samp(src, sw, x1, y1, ch);
                const top = a + (b - a) * tx;
                const bot = c + (d - c) * tx;
                const v = top + (bot - top) * ty;
                dst[(@as(usize, y) * @as(usize, dw) + @as(usize, x)) * 3 + ch] = @intFromFloat(@min(@max(v, 0), 255) + 0.5);
            }
        }
    }
}

fn samp(src: []const u8, sw: u32, x: u32, y: u32, ch: u32) f32 {
    return @floatFromInt(src[(@as(usize, y) * @as(usize, sw) + @as(usize, x)) * 3 + ch]);
}

// ── The perceptual bar ────────────────────────────────────────────────────────
//
// The oracle above proves the projection IS the reference's. It cannot prove
// the reference's projection resembles the video, so the live tests
// (`minimax_h3_vae.zig`, `ltx_video.zig`) VAE-decode a real latent, box-average
// the decoded frame down to the latent grid and correlate it against this
// file's projection of the same latent. `goldenAngleControlMap` is the arm that
// makes that bar load-bearing.

/// Box-average an RGB8 image down to `dw x dh`. A decoded frame is 16x (H3) or
/// 32x (LTX) the latent grid, and the preview is a per-latent-pixel value, so
/// AVERAGING is the comparison — point sampling would score the fit against
/// whichever texel it happened to land on.
pub fn boxDownsampleRgb(allocator: std.mem.Allocator, src: []const u8, sw: u32, sh: u32, dw: u32, dh: u32) ![]u8 {
    if (sw == 0 or sh == 0 or dw == 0 or dh == 0) return error.BadLatentShape;
    if (src.len < @as(usize, sw) * @as(usize, sh) * 3) return error.BadLatentShape;
    const out = try allocator.alloc(u8, @as(usize, dw) * @as(usize, dh) * 3);
    var dy: u32 = 0;
    while (dy < dh) : (dy += 1) {
        const y0 = (dy * sh) / dh;
        const y1 = @max(y0 + 1, ((dy + 1) * sh) / dh);
        var dx: u32 = 0;
        while (dx < dw) : (dx += 1) {
            const x0 = (dx * sw) / dw;
            const x1 = @max(x0 + 1, ((dx + 1) * sw) / dw);
            var acc = [3]f64{ 0, 0, 0 };
            var n: f64 = 0;
            var y = y0;
            while (y < y1) : (y += 1) {
                var x = x0;
                while (x < x1) : (x += 1) {
                    for (0..3) |k| acc[k] += @floatFromInt(src[(@as(usize, y) * @as(usize, sw) + @as(usize, x)) * 3 + k]);
                    n += 1;
                }
            }
            for (0..3) |k| out[(@as(usize, dy) * @as(usize, dw) + @as(usize, dx)) * 3 + k] =
                @intFromFloat(@min(255.0, acc[k] / n + 0.5));
        }
    }

    return out;
}

/// Pearson correlation over every byte of two same-size RGB8 buffers. CENTERED,
/// so it is invariant to the per-frame brightness/contrast offset a preview and
/// a decode will never share — and invariant to the mean/std stretch the retired
/// projection applied, which is why the control arm needs no dead code path.
pub fn rgbCorrelation(a: []const u8, b: []const u8) f64 {
    if (a.len == 0 or a.len != b.len) return 0;
    const n: f64 = @floatFromInt(a.len);
    var sa: f64 = 0;
    var sb: f64 = 0;
    for (a, b) |x, y| {
        sa += @floatFromInt(x);
        sb += @floatFromInt(y);
    }
    const ma = sa / n;
    const mb = sb / n;
    var num: f64 = 0;
    var va: f64 = 0;
    var vb: f64 = 0;
    for (a, b) |x, y| {
        const dx = @as(f64, @floatFromInt(x)) - ma;
        const dy = @as(f64, @floatFromInt(y)) - mb;
        num += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    if (va == 0 or vb == 0) return 0;

    return num / @sqrt(va * vb);
}

/// Correlation over the CHROMA residual: each pixel's channel minus that
/// pixel's own mean. Luminance is the easy half — ANY sum of latent channels
/// tracks brightness, and on LTX's 128 channels even the golden-angle control
/// scores 0.63 on full RGB. What a fit gets right and a hue wheel cannot is
/// WHICH COLOUR, and that is all that survives here. Needs no centering: the
/// residuals of a pixel sum to zero by construction, so the global mean is 0.
pub fn rgbChromaCorrelation(a: []const u8, b: []const u8) f64 {
    if (a.len == 0 or a.len != b.len or a.len % 3 != 0) return 0;
    var num: f64 = 0;
    var va: f64 = 0;
    var vb: f64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 3) {
        var pa: [3]f64 = undefined;
        var pb: [3]f64 = undefined;
        var sa: f64 = 0;
        var sb: f64 = 0;
        for (0..3) |k| {
            pa[k] = @floatFromInt(a[i + k]);
            pb[k] = @floatFromInt(b[i + k]);
            sa += pa[k];
            sb += pb[k];
        }
        for (0..3) |k| {
            const dx = pa[k] - sa / 3.0;
            const dy = pb[k] - sb / 3.0;
            num += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
    }
    if (va == 0 or vb == 0) return 0;

    return num / @sqrt(va * vb);
}

/// Deterministic content for the perceptual bar: one pseudo-random RGB per
/// LATENT CELL (`block` pixels square), returned as `[3, px, px]` f32 in
/// [-1, 1], channel-major — the shape both VAE encoders take.
///
/// Smooth ramps are NOT enough. A monotone gradient is nearly rank-1, and
/// summing 128 latent channels at uniform magnitude reproduces one of those
/// about as well as a fit does: on LTX the golden-angle control scored chroma
/// 0.79 against the fit's 0.94 on ramps, which is no discrimination at all.
/// One independent colour per cell makes the bar 3 x cells real predictions,
/// and the control collapses where it should.
pub fn perceptualTestFrame(allocator: std.mem.Allocator, px: u32, block: u32) ![]f32 {
    if (px == 0 or block == 0 or px % block != 0) return error.BadLatentShape;
    const cells = px / block;
    const out = try allocator.alloc(f32, 3 * @as(usize, px) * @as(usize, px));
    var x: u32 = 0x9E3779B9;
    var cy: u32 = 0;
    while (cy < cells) : (cy += 1) {
        var cx: u32 = 0;
        while (cx < cells) : (cx += 1) {
            var rgb: [3]f32 = undefined;
            for (&rgb) |*v| {
                x = x *% 1664525 +% 1013904223;
                // 0.1 .. 0.9 in the unit interval, so nothing saturates.
                v.* = @as(f32, @floatFromInt(x >> 8)) / 16777216.0 * 0.8 + 0.1;
            }
            var y = cy * block;
            while (y < (cy + 1) * block) : (y += 1) {
                var w = cx * block;
                while (w < (cx + 1) * block) : (w += 1) {
                    for (0..3) |c| out[c * @as(usize, px) * px + @as(usize, y) * px + w] = rgb[c] * 2.0 - 1.0;
                }
            }
        }
    }

    return out;
}

/// The projection this file used to ship (issue #208 review): channel `i` gets
/// a golden-angle hue, every channel the same magnitude. Kept ONLY as the
/// control arm of the perceptual bar — a preview that scores no better than
/// this is a colourful noise blob, whatever its provenance.
pub fn goldenAngleControlMap(comptime c: usize) Map {
    const factors = comptime blk: {
        @setEvalBranchQuota(100000);
        const inv_c = 1.0 / @sqrt(@as(f32, @floatFromInt(c)));
        var f: [c][3]f32 = undefined;
        for (&f, 0..) |*row, i| {
            const angle = @as(f32, @floatFromInt(i)) * 2.399963229728653;
            row[0] = @cos(angle) * inv_c;
            row[1] = @cos(angle + 2.0943951023931953) * inv_c;
            row[2] = @cos(angle + 4.1887902047863905) * inv_c;
        }
        break :blk f;
    };

    return .{ .name = "GoldenAngleControl", .factors = &factors, .bias = .{ 0, 0, 0 } };
}

/// The pinned latent `tests/dump_latent_rgb_factors.py` hands the reference.
/// Integer LCG then f32-only arithmetic: a 24-bit numerator over 2^24 is exact
/// in f32, so every step below is one IEEE-754 single op and both languages
/// land on identical bits.
fn pinnedLatent(out: []f32, salt: u32) void {
    var x: u32 = 0x9E3779B9 ^ salt;
    for (out) |*v| {
        x = x *% 1664525 +% 1013904223;
        v.* = @as(f32, @floatFromInt(x >> 8)) / 16777216.0 * 3.0 - 1.5;
    }
}

test "temporalIndices: one mid slice, I2V uses frame 0" {
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{5}, temporalIndices(&buf, 11, 1, false));
    try std.testing.expectEqualSlices(u32, &.{0}, temporalIndices(&buf, 11, 1, true));
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 10 }, temporalIndices(&buf, 11, 3, false));
}

test "Opts.normalize clamps frames and max_side, keeps 0 = native" {
    const a = (Opts{ .frames = 0, .max_side = 0 }).normalize();
    try std.testing.expectEqual(@as(u32, 1), a.frames);
    try std.testing.expectEqual(@as(u32, 0), a.max_side);
    const b = (Opts{ .frames = 99, .max_side = 99999 }).normalize();
    try std.testing.expectEqual(Opts.max_frames, b.frames);
    try std.testing.expectEqual(Opts.max_side_cap, b.max_side);
}

test "the two published maps are the shapes our two video backends produce" {
    try std.testing.expectEqual(@as(u32, 128), ltx_av.channels());
    try std.testing.expectEqual(@as(u32, 24), minimax_h3.channels());
    try std.testing.expectEqualStrings("LTXAV", ltx_av.name);
    try std.testing.expectEqualStrings("MiniMaxH3Video", minimax_h3.name);
    // A fit is not a hue wheel: the channels carry very different weight. A
    // golden-angle / uniform-magnitude table would fail this.
    var min_norm: f32 = std.math.floatMax(f32);
    var max_norm: f32 = 0;
    for (minimax_h3.factors) |f| {
        const n = @abs(f[0]) + @abs(f[1]) + @abs(f[2]);
        min_norm = @min(min_norm, n);
        max_norm = @max(max_norm, n);
    }
    try std.testing.expect(max_norm > min_norm * 20.0);
}

test "latent2rgb fixtures: the projection IS the reference's, per map" {
    const a = std.testing.allocator;
    const fixture = @embedFile("fixtures/latent_rgb.json");
    var parsed = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    // (1) the tables in src/latent_rgb.zig ARE the generated golden numbers.
    const maps = root.get("maps").?.object;
    inline for (.{ .{ "ltx_av", ltx_av }, .{ "minimax_h3", minimax_h3 } }) |pair| {
        const m = maps.get(pair[0]).?.object;
        const map: Map = pair[1];
        try std.testing.expectEqualStrings(m.get("class").?.string, map.name);
        try std.testing.expectEqual(@as(i64, map.channels()), m.get("channels").?.integer);
        const rows = m.get("factors").?.array;
        try std.testing.expectEqual(map.factors.len, rows.items.len);
        for (rows.items, map.factors) |row, want| {
            for (row.array.items, want) |got, w| {
                try std.testing.expectEqual(w, @as(f32, @floatCast(got.float)));
            }
        }
        for (m.get("bias").?.array.items, map.bias) |got, w| {
            try std.testing.expectEqual(w, @as(f32, @floatCast(got.float)));
        }
    }

    // (2) our slice → RGB reproduces the reference's own output on a pinned
    // latent that has MORE pixels than the map has channels, so no combination
    // of wrong coefficients can cancel across the case.
    for (root.get("cases").?.array.items) |case_val| {
        const case = case_val.object;
        const name = case.get("map").?.string;
        const map: Map = if (std.mem.eql(u8, name, "ltx_av")) ltx_av else minimax_h3;
        const c: u32 = @intCast(case.get("c").?.integer);
        const t: u32 = @intCast(case.get("t").?.integer);
        const h: u32 = @intCast(case.get("h").?.integer);
        const w: u32 = @intCast(case.get("w").?.integer);
        const ti: u32 = @intCast(case.get("ti").?.integer);
        const salt: u32 = @intCast(case.get("salt").?.integer);
        try std.testing.expectEqual(map.channels(), c);
        try std.testing.expect(@as(usize, h) * @as(usize, w) > map.factors.len);

        const lat = try a.alloc(f32, @as(usize, c) * t * h * w);
        defer a.free(lat);
        pinnedLatent(lat, salt);

        const rgb = try a.alloc(u8, @as(usize, h) * w * 3);
        defer a.free(rgb);
        latentSliceToRgb(map, lat, t, h, w, ti, rgb);

        // The float projection: same math, different reduction order than
        // torch's BLAS matmul, so this is a tolerance and not bit equality.
        const want_f32 = case.get("rgb_f32").?.array;
        try std.testing.expectEqual(rgb.len, want_f32.items.len);
        const hw: usize = @as(usize, h) * w;
        const thw: usize = @as(usize, t) * hw;
        var max_abs: f32 = 0;
        for (0..hw) |pix| {
            var acc = map.bias;
            for (map.factors, 0..) |f, ci| {
                const v = lat[ci * thw + @as(usize, ti) * hw + pix];
                acc[0] += v * f[0];
                acc[1] += v * f[1];
                acc[2] += v * f[2];
            }
            for (0..3) |k| {
                const want: f32 = @floatCast(want_f32.items[pix * 3 + k].float);
                max_abs = @max(max_abs, @abs(acc[k] - want));
            }
        }
        try std.testing.expect(max_abs < 1e-5);

        // The quantized image: reproducing the reference's clamp + truncate.
        // Only a value landing within one reduction-order epsilon of an integer
        // boundary may differ, and never by more than 1.
        const want_u8 = case.get("rgb_u8").?.array;
        try std.testing.expectEqual(rgb.len, want_u8.items.len);
        var off_by_one: usize = 0;
        for (rgb, want_u8.items) |got, want_val| {
            const want: i32 = @intCast(want_val.integer);
            const diff = @as(i32, got) - want;
            try std.testing.expect(diff >= -1 and diff <= 1);
            if (diff != 0) off_by_one += 1;
        }
        // A wrong map is not "a few boundary pixels" — it is most of them.
        try std.testing.expect(off_by_one * 100 < rgb.len);
    }
}

test "latentSliceToRgb: the temporal index selects its own frame" {
    const a = std.testing.allocator;
    const map = minimax_h3;
    const c = map.channels();
    const t: u32 = 4;
    const h: u32 = 2;
    const w: u32 = 2;
    const hw: usize = h * w;
    const lat = try a.alloc(f32, @as(usize, c) * t * hw);
    defer a.free(lat);
    // Frame f is the constant f * 0.25 in every channel.
    for (0..c) |ci| {
        for (0..t) |f| {
            for (0..hw) |p| lat[ci * @as(usize, t) * hw + f * hw + p] = @as(f32, @floatFromInt(f)) * 0.25;
        }
    }
    var prev: ?[3]u8 = null;
    for (0..t) |f| {
        const rgb = try a.alloc(u8, hw * 3);
        defer a.free(rgb);
        latentSliceToRgb(map, lat, t, h, w, @intCast(f), rgb);
        // Constant frame → every pixel identical.
        for (1..hw) |p| try std.testing.expectEqualSlices(u8, rgb[0..3], rgb[p * 3 ..][0..3]);
        // And a DIFFERENT frame reads differently, so the index is honoured.
        if (prev) |pv| try std.testing.expect(!std.mem.eql(u8, &pv, rgb[0..3]));
        prev = .{ rgb[0], rgb[1], rgb[2] };
    }
}

test "jpegFromLatent: preview_frames>1 is a filmstrip in temporal order" {
    const a = std.testing.allocator;
    const map = minimax_h3;
    const c = map.channels();
    const t: u32 = 4;
    const h: u32 = 4;
    const w: u32 = 4;
    const lat = try a.alloc(f32, @as(usize, c) * t * h * w);
    defer a.free(lat);
    @memset(lat, 0);
    const enc = try jpegFromLatent(a, map, lat, t, h, w, .{ .frames = 4, .max_side = 0 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expectEqual(@as(u32, 16), enc.w);
    try std.testing.expectEqual(@as(u32, 4), enc.h);

    // The strip's tiles are the frames in order: give frame f a distinct
    // constant and check the pre-JPEG plane the strip is built from.
    for (0..c) |ci| {
        for (0..t) |f| {
            for (0..@as(usize, h) * w) |p|
                lat[ci * @as(usize, t) * h * w + f * @as(usize, h) * w + p] = @as(f32, @floatFromInt(f)) * 0.3;
        }
    }
    var idx_buf: [Opts.max_frames]u32 = undefined;
    const idx = temporalIndices(&idx_buf, t, 4, false);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, idx);
    var seen: [4][3]u8 = undefined;
    for (idx, 0..) |ti, fi| {
        const rgb = try a.alloc(u8, @as(usize, h) * w * 3);
        defer a.free(rgb);
        latentSliceToRgb(map, lat, t, h, w, ti, rgb);
        seen[fi] = .{ rgb[0], rgb[1], rgb[2] };
    }
    for (1..4) |i| try std.testing.expect(!std.mem.eql(u8, &seen[i - 1], &seen[i]));
}

test "jpegFromLatent: 24-ch H3-shaped volume yields a JPEG" {
    const a = std.testing.allocator;
    const t: u32 = 5;
    const h: u32 = 8;
    const w: u32 = 12;
    const lat = try a.alloc(f32, @as(usize, minimax_h3.channels()) * t * h * w);
    defer a.free(lat);
    pinnedLatent(lat, 7);
    const enc = try jpegFromLatent(a, minimax_h3, lat, t, h, w, .{ .enabled = true, .frames = 1, .max_side = 256 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expect(enc.jpeg.len > 16);
    try std.testing.expectEqual(@as(u8, 0xFF), enc.jpeg[0]);
    try std.testing.expectEqual(@as(u8, 0xD8), enc.jpeg[1]);
    try std.testing.expectEqual(@as(u32, w), enc.w);
    try std.testing.expectEqual(@as(u32, h), enc.h);
    try std.testing.expectEqualStrings("image/jpeg", enc.mime);
}

test "jpegFromLatent: 128-ch LTX-shaped volume yields a JPEG" {
    const a = std.testing.allocator;
    const t: u32 = 2;
    const h: u32 = 4;
    const w: u32 = 6;
    const lat = try a.alloc(f32, @as(usize, ltx_av.channels()) * t * h * w);
    defer a.free(lat);
    pinnedLatent(lat, 11);
    const enc = try jpegFromLatent(a, ltx_av, lat, t, h, w, .{ .max_side = 0 }, true);
    defer a.free(enc.jpeg);
    try std.testing.expect(enc.jpeg[0] == 0xFF and enc.jpeg[1] == 0xD8);
}

test "jpegFromLatent: max_side downscales the long edge" {
    const a = std.testing.allocator;
    const t: u32 = 1;
    const h: u32 = 32;
    const w: u32 = 64;
    const lat = try a.alloc(f32, @as(usize, minimax_h3.channels()) * t * h * w);
    defer a.free(lat);
    @memset(lat, 0.25);
    const enc = try jpegFromLatent(a, minimax_h3, lat, t, h, w, .{ .max_side = 16 }, false);
    defer a.free(enc.jpeg);
    try std.testing.expectEqual(@as(u32, 16), enc.w);
    try std.testing.expectEqual(@as(u32, 8), enc.h);
}

test "jpegFromLatent: a latent whose channel count is not the map's is refused" {
    const a = std.testing.allocator;
    const lat = try a.alloc(f32, 24 * 2 * 4 * 4);
    defer a.free(lat);
    @memset(lat, 0);
    // 24 channels of data handed the 128-channel map is a short buffer, not a
    // silently wrong picture.
    try std.testing.expectError(error.BadLatentShape, jpegFromLatent(a, ltx_av, lat, 2, 4, 4, .{}, false));
}

test "formatProgressJson: default event has no preview key" {
    const a = std.testing.allocator;
    const s = try formatProgressJson(a, "Generating", 8, 30, null);
    defer a.free(s);
    try std.testing.expectEqualStrings(
        "{\"type\":\"progress\",\"stage\":\"Generating\",\"step\":8,\"total\":30}",
        s,
    );
    var parsed = try std.json.parseFromSlice(std.json.Value, a, s, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.object.get("preview") == null);
}

test "formatProgressJson: preview event is parseable JSON with JPEG b64" {
    const a = std.testing.allocator;
    const jpg = [_]u8{ 0xFF, 0xD8, 0xFF, 0x00 };
    const enc = Encoded{ .jpeg = @constCast(&jpg), .w = 256, .h = 144 };
    const s = try formatProgressJson(a, "Generating", 8, 30, enc);
    defer a.free(s);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, s, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try std.testing.expectEqualStrings("progress", o.get("type").?.string);
    try std.testing.expectEqual(@as(i64, 8), o.get("step").?.integer);
    try std.testing.expectEqual(@as(i64, 30), o.get("total").?.integer);
    try std.testing.expectEqualStrings("image/jpeg", o.get("mime").?.string);
    try std.testing.expectEqual(@as(i64, 256), o.get("w").?.integer);
    try std.testing.expectEqual(@as(i64, 144), o.get("h").?.integer);
    const b64 = o.get("preview").?.string;
    const raw = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(b64));
    defer a.free(raw);
    try std.base64.standard.Decoder.decode(raw, b64);
    try std.testing.expectEqualSlices(u8, &jpg, raw);
}

test "boxDownsampleRgb averages its source block, never point-samples it" {
    const a = std.testing.allocator;
    // 4x2 where each 2x2 block is (0, 255) side by side: the average is 127/128.
    var src: [4 * 2 * 3]u8 = undefined;
    for (0..2) |y| {
        for (0..4) |x| {
            const v: u8 = if (x % 2 == 0) 0 else 255;
            for (0..3) |k| src[(y * 4 + x) * 3 + k] = v;
        }
    }
    const out = try boxDownsampleRgb(a, &src, 4, 2, 2, 1);
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, 6), out.len);
    for (out) |v| try std.testing.expect(v == 127 or v == 128);
}

test "rgbCorrelation: 1 on itself, invariant to brightness and contrast, -1 inverted" {
    const a = [_]u8{ 10, 40, 90, 200, 130, 60, 20, 250, 5 };
    var scaled: [a.len]u8 = undefined;
    var inverted: [a.len]u8 = undefined;
    for (a, 0..) |v, i| {
        scaled[i] = @intCast(@min(255, @as(u32, v) / 2 + 40));
        inverted[i] = 255 - v;
    }
    try std.testing.expect(rgbCorrelation(&a, &a) > 0.9999);
    try std.testing.expect(rgbCorrelation(&a, &scaled) > 0.99);
    try std.testing.expect(rgbCorrelation(&a, &inverted) < -0.9999);
    // A flat buffer has no structure to correlate with: 0, never NaN.
    var flat: [a.len]u8 = undefined;
    @memset(&flat, 128);
    try std.testing.expectEqual(@as(f64, 0), rgbCorrelation(&a, &flat));
}

test "rgbChromaCorrelation ignores brightness and convicts a wrong hue" {
    // The exact situation the LTX control arm produces: two images that share a
    // luminance ramp (so full-RGB correlation reads high whatever the colour)
    // and disagree only on hue.
    const n = 8;
    var a: [n * 3]u8 = undefined;
    var wrong_hue: [n * 3]u8 = undefined;
    var brighter: [n * 3]u8 = undefined;
    var grey: [n * 3]u8 = undefined;
    for (0..n) |p| {
        // The ramp and the lift both stay inside 0..255: a CLAMP would make the
        // brightness lift non-uniform and stop being the thing under test.
        const lum: i32 = 40 + @divTrunc(@as(i32, @intCast(p)) * 160, n - 1);
        const bias: i32 = 12;
        const av = [3]i32{ lum + bias, lum - bias, lum };
        const bv = [3]i32{ lum - bias, lum + bias, lum }; // R and G traded
        for (0..3) |k| {
            a[p * 3 + k] = @intCast(av[k]);
            wrong_hue[p * 3 + k] = @intCast(bv[k]);
            brighter[p * 3 + k] = @intCast(av[k] + 20);
            grey[p * 3 + k] = @intCast(lum);
        }
    }
    try std.testing.expect(rgbChromaCorrelation(&a, &a) > 0.9999);
    // A uniform brightness lift is invisible to chroma.
    try std.testing.expect(rgbChromaCorrelation(&a, &brighter) > 0.9999);
    // The whole point: the shared ramp keeps full RGB high, chroma convicts.
    try std.testing.expect(rgbCorrelation(&a, &wrong_hue) > 0.9);
    try std.testing.expect(rgbChromaCorrelation(&a, &wrong_hue) < -0.9);
    // No chroma at all → 0, never NaN.
    try std.testing.expectEqual(@as(f64, 0), rgbChromaCorrelation(&a, &grey));
}

test "perceptualTestFrame: one independent colour per latent cell, in range" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.BadLatentShape, perceptualTestFrame(a, 12, 5));
    const px: u32 = 8;
    const block: u32 = 4;
    const f = try perceptualTestFrame(a, px, block);
    defer a.free(f);
    try std.testing.expectEqual(@as(usize, 3 * px * px), f.len);
    for (f) |v| try std.testing.expect(v >= -0.81 and v <= 0.81);
    // A cell is CONSTANT (so the decode's block average is that colour) and its
    // neighbours are not it (so there is something per-cell to predict).
    var seen: [4][3]f32 = undefined;
    for (0..2) |cy| {
        for (0..2) |cx| {
            const y0 = cy * block;
            const x0 = cx * block;
            for (0..3) |c| {
                const want = f[c * px * px + y0 * px + x0];
                for (0..block) |dy| {
                    for (0..block) |dx| {
                        try std.testing.expectEqual(want, f[c * px * px + (y0 + dy) * px + x0 + dx]);
                    }
                }
                seen[cy * 2 + cx][c] = want;
            }
        }
    }
    for (0..4) |i| {
        for (i + 1..4) |j| try std.testing.expect(!std.mem.eql(f32, &seen[i], &seen[j]));
    }
    // Deterministic: same call, same bytes.
    const g = try perceptualTestFrame(a, px, block);
    defer a.free(g);
    try std.testing.expectEqualSlices(f32, f, g);
}

test "goldenAngleControlMap is the retired projection: uniform magnitude, no bias" {
    const ctrl = goldenAngleControlMap(24);
    try std.testing.expectEqual(@as(u32, 24), ctrl.channels());
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0 }, &ctrl.bias);
    // Every channel pulls equally hard — the property that makes it a hue wheel
    // and not a fit, and the exact inverse of the `max > 20x min` fit assertion.
    var min_norm: f32 = std.math.floatMax(f32);
    var max_norm: f32 = 0;
    for (ctrl.factors) |f| {
        const n = @sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
        min_norm = @min(min_norm, n);
        max_norm = @max(max_norm, n);
    }
    try std.testing.expect(max_norm < min_norm * 1.001);
    // And it is NOT what we ship.
    try std.testing.expect(!std.mem.eql(u8, ctrl.name, minimax_h3.name));
}

test {
    _ = @import("jpeg.zig");
}
