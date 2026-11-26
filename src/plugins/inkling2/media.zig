//! Inkling 2's media hooks (`abi.MediaHooks`): an image as mlx-lm's `models/inkling2/processor.py` prepares it. The
//! reference feeds an image at its own size: padded right and down to whole blocks (one row each) with -1 on the
//! 0..255 scale, then mean and std 0.5; its run is its rows of the image placeholder. An image past the engine's
//! pixel budget is scaled down to it first (Pillow bicubic), which the reference has no rule for.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
const config = @import("config.zig");

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;
const C = 3;
/// A padded pixel: -1 on the 0..255 scale, normalized.
const pad: f32 = (-1.0 * (1.0 / 255.0) - 0.5) / 0.5;

pub fn prepare(p: *const abi.Parsed, in: *const abi.MediaInput, budget: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    const v = config.of(p).vision orelse return Verdict.refuse(.unsupported, "inkling2: this pack has no image tower");
    if (in.kind != .image) return Verdict.refuse(.unsupported, "inkling2: images only");
    if (in.width == 0 or in.height == 0 or in.rgb.len != @as(usize, in.width) * in.height * C) return Verdict.refuse(.failed, "inkling2: an image is one RGB frame");
    const size = fitted(in.height, in.width, budget);
    const plane = @as(usize, size.h) * size.w;
    const chw = gpa.alloc(f32, C * plane) catch return Verdict.refuse(.failed, "inkling2: out of memory");
    defer gpa.free(chw);
    if (size.h == in.height and size.w == in.width) {
        for (0..plane) |i| for (0..C) |c| {
            chw[c * plane + i] = (@as(f32, @floatFromInt(in.rgb.slice()[i * C + c])) * (1.0 / 255.0) - 0.5) / 0.5;
        };
    } else qwen_vision.resizeRgbNormalizedChw(gpa, chw, in.rgb.slice(), in.height, in.width, size.h, size.w, .bicubic) catch return Verdict.refuse(.failed, "inkling2: out of memory");
    const rows = (size.h + v.block() - 1) / v.block();
    const cols = (size.w + v.block() - 1) / v.block();
    const feat: usize = C * v.patch * v.patch;
    const patches: usize = @as(usize, rows) * cols * v.merge * v.merge;
    const values = gpa.alloc(f32, patches * feat) catch return Verdict.refuse(.failed, "inkling2: out of memory");
    defer gpa.free(values);
    layout(values, chw, size.h, size.w, &v);
    out.* = .{
        .kind = .image,
        .rows = rows * cols,
        .grid = .{ 1, rows * v.merge, cols * v.merge },
        .pixels = .{ .ctx = mlx.mlx_array_new_data(values.ptr, &[_]c_int{ @intCast(patches), @intCast(feat) }, 2, .float32).ctx },
    };
    return .ok;
}

/// The image's size, or the largest of its aspect within `budget` pixels.
pub fn fitted(height: u32, width: u32, budget: u64) qwen_vision.Resized {
    const area = @as(u64, height) * width;
    if (area <= budget) return .{ .h = height, .w = width };
    const scale = @sqrt(@as(f64, @floatFromInt(budget)) / @as(f64, @floatFromInt(area)));
    return .{ .h = @max(1, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(height)) * scale)))), .w = @max(1, @as(u32, @intFromFloat(@floor(@as(f64, @floatFromInt(width)) * scale)))) };
}

/// A normalized CHW image as the tower's patches: merge block by merge block (blocks in raster order), a block's
/// patches in raster order, a patch's features its pixels in raster order with the channel innermost; pixels past
/// the image are the padding.
pub fn layout(values: []f32, chw: []const f32, height: u32, width: u32, v: *const config.Vision) void {
    const plane = @as(usize, height) * width;
    const p = v.patch;
    const m = v.merge;
    const cols = (width + v.block() - 1) / v.block();
    const feat: usize = C * p * p;
    for (0..values.len / feat) |i| {
        const block = i / (m * m);
        const y0 = (block / cols * m + i / m % m) * p;
        const x0 = (block % cols * m + i % m) * p;
        for (0..p) |py| for (0..p) |px| for (0..C) |c| {
            const y = y0 + py;
            const x = x0 + px;
            values[i * feat + (py * p + px) * C + c] = if (y < height and x < width) chw[c * plane + y * width + x] else pad;
        };
    }
}

pub fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *@import("plugin.zig").Loaded = @ptrCast(@alignCast(mp));
    const t = &(m.tower orelse return Verdict.refuse(.unsupported, "inkling2: this pack has no image tower"));
    const rows = t.forward(.{ .ctx = item.pixels.ctx }, item.grid[1], item.grid[2]) catch return Verdict.refuse(.failed, "inkling2: the image tower failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

// ── tests ──

const testing = std.testing;

test "inkling2 media: an image is fed at its own size, padded to whole blocks with -1 on the 0..255 scale" {
    const v: config.Vision = .{ .patch = 2, .merge = 2 };
    // 5 x 6 pixels in blocks of 4: 2 x 2 blocks, 16 patches of 12 features.
    const h = 5;
    const w = 6;
    var chw: [C * h * w]f32 = undefined;
    for (&chw, 0..) |*x, i| x.* = @floatFromInt(i);
    var values: [16 * 12]f32 = undefined;
    layout(&values, &chw, h, w, &v);
    const at = struct {
        fn f(c: usize, y: usize, x: usize) f32 {
            return @floatFromInt(c * h * w + y * w + x);
        }
    }.f;
    // Patch 0 is the image's top-left 2 x 2 pixels, channel innermost.
    try testing.expectEqualSlices(f32, &.{ at(0, 0, 0), at(1, 0, 0), at(2, 0, 0), at(0, 0, 1), at(1, 0, 1), at(2, 0, 1), at(0, 1, 0), at(1, 1, 0), at(2, 1, 0), at(0, 1, 1), at(1, 1, 1), at(2, 1, 1) }, values[0..12]);
    // Patch 3 is the first block's bottom-right; patch 4 opens the second block at column 4.
    try testing.expectEqual(at(0, 2, 2), values[3 * 12]);
    try testing.expectEqual(at(0, 0, 4), values[4 * 12]);
    // Patch 5 starts at column 6, past the image; patch 10 (the third block's lower-left) at row 6.
    try testing.expectEqual(pad, values[5 * 12]);
    try testing.expectEqual(pad, values[10 * 12]);
    // Patch 8 (row 4): its first pixel row is the image's last, its second padding.
    try testing.expectEqual(at(0, 4, 0), values[8 * 12]);
    try testing.expectEqual(pad, values[8 * 12 + 6]);
    try testing.expectApproxEqAbs(@as(f32, -1.0078431), pad, 1e-6);
}

test "inkling2 media: an image past the budget keeps its aspect within it" {
    try testing.expectEqual(qwen_vision.Resized{ .h = 300, .w = 400 }, fitted(300, 400, 1 << 20));
    const s = fitted(3000, 4000, 1 << 20);
    try testing.expect(@as(u64, s.h) * s.w <= 1 << 20 and s.h == 886 and s.w == 1182);
}
