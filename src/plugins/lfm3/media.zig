//! LFM3's media hooks (`abi.MediaHooks`). A speech clip (16 kHz mono) is framed for the encoder (`audio.zig`); its run
//! is its rows of `<|audio|>`. Images and clips as mlx-vlm's `lfm3_vl/processing_lfm3_vl.py` prepares and lays out an
//! item: an image takes Qwen3-VL's processor (smart resize to the patch-merge grid, bicubic, mean/std 0.5, the still
//! duplicated across the temporal patch); a clip takes LFM3's own rule (one size for every frame, from the whole clip's
//! pixel bounds; the last frame repeats to fill a temporal group). A clip's run is, per frame group,
//! `<t seconds><|vision_start|>` + its rows + `<|vision_end|>`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const config = @import("config.zig");
const audio = @import("audio.zig");

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;
const C = 3;

pub fn prepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    if (in.kind == .audio) return prepareSpeech(p, in, out);
    const v = config.of(p).vision orelse return Verdict.refuse(.unsupported, "lfm3: this pack has no vision tower");
    const frames: usize = if (in.kind == .image) 1 else in.frames;
    if (in.kind == .audio or frames == 0 or in.width == 0 or in.height == 0 or
        in.rgb.len != frames * @as(usize, in.width) * in.height * C)
        return Verdict.refuse(.failed, "lfm3: images and clips of RGB frames only");
    const cap: u32 = @intCast(@min(budget, std.math.maxInt(u32)));
    const size = if (in.kind == .image)
        qwen_vision.smartResizeImage(in.height, in.width, v.patch * v.merge, v.image_min_pixels, @min(v.image_max_pixels, cap))
    else
        clipSize(in.height, in.width, frames, v, cap) orelse return Verdict.refuse(.failed, "lfm3: the clip's frames have no grid");
    const pv = pixelValues(in.rgb.slice(), in.height, in.width, frames, size, v) catch return Verdict.refuse(.failed, "lfm3: out of memory");
    defer gpa.free(pv.values);
    const gh = size.h / v.patch;
    const gw = size.w / v.patch;
    const feat: c_int = @intCast(C * v.temporal * v.patch * v.patch);
    out.* = .{
        .kind = in.kind,
        .rows = @intCast(pv.groups * (gh / v.merge) * (gw / v.merge)),
        .grid = .{ pv.groups, gh, gw },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(pv.values.ptr, &[_]c_int{ @intCast(pv.values.len / @as(usize, @intCast(feat))), feat }, 2, .float32).ctx },
        .fps = if (in.kind == .video) (if (in.fps > 0) in.fps else v.fps) else 0,
        .frames = @intCast(frames),
    };
    return .ok;
}

/// A clip right-padded to its frames' span (`pixels`, f32 samples); `grid[0]` is its feature frames.
fn prepareSpeech(p: *const abi.Parsed, in: *const abi.MediaInput, out: *abi.MediaItem) Verdict {
    if (config.of(p).audio == null) return Verdict.refuse(.unsupported, "lfm3: this pack has no speech encoder");
    if (in.sample_rate != audio.SAMPLE_RATE or in.samples.len == 0) return Verdict.refuse(.failed, "lfm3: speech is 16 kHz mono samples");
    const wave = audio.padded(gpa, in.samples.slice()) catch return Verdict.refuse(.failed, "lfm3: out of memory");
    defer gpa.free(wave);
    const n = audio.frames(in.samples.len);
    out.* = .{
        .kind = .audio,
        .rows = @intCast(audio.rows(n)),
        .grid = .{ @intCast(n), 0, 0 },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(wave.ptr, &[_]c_int{@intCast(wave.len)}, 1, .float32).ctx },
    };
    return .ok;
}

/// LFM3's `Lfm3VlVideoProcessor._process_one` size: each side snapped to the patch-merge factor, then the whole
/// clip (frames rounded to temporal groups) scaled into the pixel bounds; a frame past `cap` pixels shrinks to it.
pub fn clipSize(height: u32, width: u32, frames: usize, v: config.Vision, cap: u32) ?qwen_vision.Resized {
    const factor: f64 = @floatFromInt(v.patch * v.merge);
    var h: f64 = @floatFromInt(height);
    var w: f64 = @floatFromInt(width);
    if (@min(h, w) < factor) {
        const scale = factor / @min(h, w);
        h = @floor(h * scale);
        w = @floor(w * scale);
    }
    if (@max(h, w) / @min(h, w) > 200) return null;
    var th = qwen_vision.roundHalfEven(h / factor) * factor;
    var tw = qwen_vision.roundHalfEven(w / factor) * factor;
    const t: f64 = @floatFromInt(frames);
    const tps: f64 = @floatFromInt(v.temporal);
    const rounded_t = qwen_vision.roundHalfEven(t / tps) * tps;
    const max: f64 = @floatFromInt(v.video_max_pixels);
    const min: f64 = @floatFromInt(v.video_min_pixels);
    if (rounded_t * th * tw > max) {
        const beta = @sqrt(t * h * w / max);
        th = @max(factor, @floor(h / beta / factor) * factor);
        tw = @max(factor, @floor(w / beta / factor) * factor);
    } else if (rounded_t * th * tw < min) {
        const beta = @sqrt(min / (t * h * w));
        th = @ceil(h * beta / factor) * factor;
        tw = @ceil(w * beta / factor) * factor;
    }
    const c: f64 = @floatFromInt(cap);
    if (th * tw > c) {
        const beta = @sqrt(th * tw / c);
        th = @max(factor, @floor(th / beta / factor) * factor);
        tw = @max(factor, @floor(tw / beta / factor) * factor);
    }
    if (th < factor or tw < factor) return null;
    return .{ .h = @intFromFloat(th), .w = @intFromFloat(tw) };
}

const Pixels = struct { values: []f32, groups: u32 };

/// Every frame resized and normalized, the last repeated to fill its temporal group, laid out group by group in
/// merge-block order with features `[C, temporal, py, px]`.
fn pixelValues(rgb: []const u8, height: u32, width: u32, frames: usize, size: qwen_vision.Resized, v: config.Vision) !Pixels {
    const plane: usize = C * @as(usize, size.h) * size.w;
    const chw = try gpa.alloc(f32, frames * plane);
    defer gpa.free(chw);
    const src: usize = C * @as(usize, height) * width;
    for (0..frames) |k|
        try qwen_vision.resizeRgbNormalizedChw(gpa, chw[k * plane ..][0..plane], rgb[k * src ..][0..src], height, width, size.h, size.w, .bicubic);
    const groups: usize = (frames + v.temporal - 1) / v.temporal;
    const per_group: usize = plane * v.temporal;
    const values = try gpa.alloc(f32, groups * per_group);
    errdefer gpa.free(values);
    var group: [8][]const f32 = undefined;
    if (v.temporal > group.len) return error.TemporalPatchTooWide;
    for (0..groups) |g| {
        for (0..v.temporal) |k| group[k] = chw[@min(g * v.temporal + k, frames - 1) * plane ..][0..plane];
        qwen_vision.buildPixelValuesVideo(values[g * per_group ..][0..per_group], group[0..v.temporal], C, size.h, size.w, v.patch, v.merge);
    }
    return .{ .values = values, .groups = @intCast(groups) };
}

pub fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *@import("plugin.zig").Loaded = @ptrCast(@alignCast(mp));
    if (item.kind == .audio) {
        const enc = &(m.speech orelse return Verdict.refuse(.unsupported, "lfm3: this pack has no speech encoder"));
        const rows = enc.forward(.{ .ctx = item.pixels.ctx }, item.grid[0]) catch return Verdict.refuse(.failed, "lfm3: the speech encoder failed");
        out.* = .{ .ctx = rows.ctx };
        return .ok;
    }
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "lfm3: this pack has no vision tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[0], item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "lfm3: the vision tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

pub fn placeholderRows(p: *const abi.Parsed, item: *const abi.MediaItem, tok: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
    const ids = p.placeholder_ids;
    if (item.kind == .audio) {
        if (item.rows > cap) return 0;
        @memset(out[0..item.rows], ids.audio);
        return item.rows;
    }
    if (item.kind == .image) {
        if (item.rows > cap) return 0;
        @memset(out[0..item.rows], ids.image);
        return item.rows;
    }
    const groups = item.grid[0];
    const v = config.of(p).vision orelse return 0;
    if (groups == 0 or item.fps <= 0 or item.frames == 0) return 0;
    const per_group = item.rows / groups;
    var n: u32 = 0;
    for (0..groups) |g| {
        var text: [48]u8 = undefined;
        const label = timestamp(&text, groupSeconds(g, v.temporal, item.frames, item.fps)) catch return 0;
        n += tok.encode(tok.ctx, abi.Str.of(label), out + n, cap - n);
        if (n + per_group + 2 > cap) return 0;
        out[n] = ids.vision_start;
        @memset(out[n + 1 .. n + 1 + per_group], ids.video);
        out[n + 1 + per_group] = ids.vision_end;
        n += per_group + 2;
    }
    return n;
}

/// The group's time as the reference stamps it: the mean of its first and last frame's seconds, the padded last group
/// reading the last real frame twice.
pub fn groupSeconds(g: usize, temporal: u32, frames: u32, fps: f32) f64 {
    const first: f64 = @floatFromInt(@min(g * temporal, frames - 1));
    const last: f64 = @floatFromInt(@min(g * temporal + temporal - 1, frames - 1));
    const rate: f64 = fps;
    return (first / rate + last / rate) / 2;
}

/// `f"<{seconds:.1f} seconds>"`: Python rounds the double's exact value, ties to even.
pub fn timestamp(buf: []u8, seconds: f64) ![]const u8 {
    const x: f128 = @as(f128, seconds) * 10;
    const fl = @floor(x);
    const frac = x - fl;
    const tenths: u64 = @intFromFloat(if (frac > 0.5 or (frac == 0.5 and @mod(fl, 2) != 0)) fl + 1 else fl);
    return std.fmt.bufPrint(buf, "<{d}.{d} seconds>", .{ tenths / 10, tenths % 10 });
}

// ── tests ──

const testing = std.testing;

test "lfm3 media: timestamps round as Python's .1f does (the exact double, ties to even)" {
    var b: [48]u8 = undefined;
    // Python: f"{x:.1f}" for 0.25, 0.75, 0.15, 0.35, 2.65, 1.0, 12.05.
    const cases = [_]struct { f64, []const u8 }{ .{ 0.25, "<0.2 seconds>" }, .{ 0.75, "<0.8 seconds>" }, .{ 0.15, "<0.1 seconds>" }, .{ 0.35, "<0.3 seconds>" }, .{ 2.65, "<2.6 seconds>" }, .{ 1.0, "<1.0 seconds>" }, .{ 12.05, "<12.1 seconds>" } };
    for (cases) |c| try testing.expectEqualStrings(c[1], try timestamp(&b, c[0]));
}

test "lfm3 media: a clip's groups are stamped at their frames' mean, the padded last group at its real frame" {
    // Five frames at 2 fps in groups of two: (0,1), (2,3), (4,4).
    try testing.expectEqual(@as(f64, 0.25), groupSeconds(0, 2, 5, 2.0));
    try testing.expectEqual(@as(f64, 1.25), groupSeconds(1, 2, 5, 2.0));
    try testing.expectEqual(@as(f64, 2.0), groupSeconds(2, 2, 5, 2.0));
}
