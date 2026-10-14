//! MiniCPM-V 4.7 canvas M-RoPE (`mrope_mode: "canvas"`).
//!
//! Port of `MiniCPMV4_7Model.get_rope_index`. A picture is laid out on a 2-D canvas measured in
//! LLM tokens: its thumbnail is stretched over the whole canvas, each detail crop sits at its
//! grid cell, and the `<image>`/`<slice>`/`</slice>`/`</image>` markers and row newlines are
//! pinned to the corners and edges they delimit. Text stays 1-D. Pure host arithmetic over the
//! expanded prompt ids, so it is tested without a model.

const std = @import("std");

pub const Markers = struct {
    slice_start: u32,
    slice_end: u32,
    image_start: u32,
    image_end: u32,
    newline: u32,

    fn contains(self: Markers, id: u32) bool {
        return id == self.slice_start or id == self.slice_end or id == self.image_start or id == self.image_end or id == self.newline;
    }
};

/// Patch grid `(h, w)` of one crop, in prompt order.
pub const Grid = [2]u32;

pub const Layout = struct {
    markers: Markers,
    image_token: u32,
    /// 0 when the checkpoint serves no video.
    video_token: u32,
    /// Patches per LLM token along each side: 4 for `16x`, 2 for `4x`.
    merge: u32,
};

pub const Result = struct {
    /// `[3 * ids.len]` i32, the T, H and W rows back to back.
    pos: []i32,
    /// Decode position of token `n` is `n + delta`.
    delta: i32,
};

const Span = struct { start: usize, end: usize };

const Frame = struct {
    modality: u8,
    thumb: Span,
    slices: std.ArrayList(Span),
    end: usize = 0,
};

const Group = struct { start: usize, frames: std.ArrayList(Frame), end: usize = 0 };

/// One past the markers that close a frame, starting at the end of its last crop.
fn frameEnd(ids: []const u32, last_crop_end: usize, markers: Markers) usize {
    var end = last_crop_end;
    while (end < ids.len and markers.contains(ids[end])) end += 1;
    return end;
}

/// `_group_visual_frames`: crops (maximal non-text runs) -> frames (a thumbnail plus its slices)
/// -> groups (frames only markers separate share one canvas budget).
fn groupFrames(allocator: std.mem.Allocator, ids: []const u32, mm: []const u8, markers: Markers) !std.ArrayList(Group) {
    var groups = std.ArrayList(Group).empty;
    errdefer freeGroups(allocator, &groups);
    var open_frame = false;
    var previous_end: usize = 0;
    var i: usize = 0;
    while (i < ids.len) {
        const modality = mm[i];
        var j = i;
        while (j < ids.len and mm[j] == modality) j += 1;
        defer i = j;
        if (modality == 0) continue;
        const crop = Span{ .start = i, .end = j };

        if (open_frame and ids[crop.start - 1] == markers.slice_start) {
            const frames = &groups.items[groups.items.len - 1].frames;
            try frames.items[frames.items.len - 1].slices.append(allocator, crop);
        } else {
            var adjacent = groups.items.len > 0;
            if (adjacent) for (ids[previous_end .. crop.start - 1]) |t| {
                if (!markers.contains(t)) adjacent = false;
            };
            if (!adjacent) try groups.append(allocator, .{ .start = crop.start - 1, .frames = .empty });
            const group = &groups.items[groups.items.len - 1];
            try group.frames.append(allocator, .{ .modality = modality, .thumb = crop, .slices = .empty });
            open_frame = true;
        }
        previous_end = crop.end;
    }
    for (groups.items) |*g| {
        for (g.frames.items) |*f| {
            f.end = frameEnd(ids, if (f.slices.items.len > 0) f.slices.items[f.slices.items.len - 1].end else f.thumb.end, markers);
        }
        g.end = g.frames.items[g.frames.items.len - 1].end;
    }
    return groups;
}

fn freeGroups(allocator: std.mem.Allocator, groups: *std.ArrayList(Group)) void {
    for (groups.items) |*g| {
        for (g.frames.items) |*f| f.slices.deinit(allocator);
        g.frames.deinit(allocator);
    }
    groups.deinit(allocator);
}

/// `torch.linspace(0, canvas - 1, n).round()` in f32: ties go to the even integer.
fn canvasCoord(canvas: u32, n: u32, i: u32) i32 {
    if (n <= 1) return 0;
    const end: f32 = @floatFromInt(canvas - 1);
    const step: f32 = end / @as(f32, @floatFromInt(n - 1));
    const v: f32 = if (i < n / 2) step * @as(f32, @floatFromInt(i)) else end - step * @as(f32, @floatFromInt(n - i - 1));
    return @intFromFloat(roundHalfEven(v));
}

fn roundHalfEven(v: f32) f32 {
    return if (@abs(v - @trunc(v)) == 0.5) 2 * @round(v / 2) else @round(v);
}

/// `get_vision_position_ids`: row-major (T=0, H, W) of one crop's LLM grid placed on a canvas,
/// written into the three rows of `pos` from `at`.
fn visionPositions(pos: [3][]i32, at: usize, start: usize, grid: Grid, merge: u32, canvas_h: u32, canvas_w: u32, h_off: u32, w_off: u32) void {
    const gh = grid[0] / merge;
    const gw = grid[1] / merge;
    const ch = if (canvas_h != 0) canvas_h else gh;
    const cw = if (canvas_w != 0) canvas_w else gw;
    var k = at;
    for (0..gh) |i| for (0..gw) |j| {
        pos[0][k] = @intCast(start);
        pos[1][k] = canvasCoord(ch, gh, @intCast(i)) + @as(i32, @intCast(h_off + start));
        pos[2][k] = canvasCoord(cw, gw, @intCast(j)) + @as(i32, @intCast(w_off + start));
        k += 1;
    };
}

fn fillRange(pos: [3][]i32, from: usize, to: usize, first: usize) void {
    for (from..to) |k| for (pos) |row| {
        row[k] = @intCast(first + (k - from));
    };
}

fn fillAll(pos: [3][]i32, from: usize, to: usize, value: usize) void {
    for (pos) |row| @memset(row[from..to], @intCast(value));
}

/// Canvas positions for one unpadded sequence. `image_grids`/`video_grids` hold every crop's
/// patch grid in prompt order (thumbnail first, then its slices).
pub fn canvasPositions(allocator: std.mem.Allocator, ids: []const u32, layout: Layout, image_grids: []const Grid, video_grids: []const Grid) !Result {
    const n = ids.len;
    const flat = try allocator.alloc(i32, 3 * n);
    errdefer allocator.free(flat);
    const pos = [3][]i32{ flat[0..n], flat[n .. 2 * n], flat[2 * n .. 3 * n] };
    for (pos) |row| for (row, 0..) |*v, k| {
        v.* = @intCast(k);
    };

    const mm = try allocator.alloc(u8, n);
    defer allocator.free(mm);
    for (ids, mm) |t, *m| m.* = if (t == layout.image_token) 1 else if (layout.video_token != 0 and t == layout.video_token) 2 else 0;

    var groups = try groupFrames(allocator, ids, mm, layout.markers);
    defer freeGroups(allocator, &groups);

    var next_grid = [3]usize{ 0, 0, 0 };
    const merge = layout.merge;
    var current_pos: usize = 0;
    var current_idx: usize = 0;
    for (groups.items) |group| {
        if (group.start > current_idx) {
            fillRange(pos, current_idx, group.start, current_pos);
            current_pos += group.start - current_idx;
        }
        var cursor = group.start;
        for (group.frames.items) |frame| {
            const grids = if (frame.modality == 1) image_grids else video_grids;
            const take = &next_grid[frame.modality];
            if (take.* >= grids.len) return error.MissingCropGrid;
            const thumb_grid = grids[take.*];
            take.* += 1;
            const frame_start = frame.thumb.start - 1;
            if (frame_start > cursor) {
                fillRange(pos, cursor, frame_start, current_pos);
                current_pos += frame_start - cursor + 1;
            }
            const frame_end = @min(frame.end, group.end);
            const origin = current_pos;
            const halo = if (origin > 0) origin - 1 else 0;

            var slice_h: u32 = 0;
            var slice_w: u32 = 0;
            var cols: u32 = 0;
            var canvas_h = thumb_grid[0] / merge;
            var canvas_w = thumb_grid[1] / merge;
            var first_slice_grid: Grid = undefined;
            const slices = frame.slices.items;
            if (slices.len > 0) {
                // A gap wider than `</slice><slice>` is the newline ending a row, which fixes the column count.
                cols = @intCast(slices.len);
                for (0..slices.len - 1) |k| {
                    if (slices[k + 1].start - slices[k].end > 2) {
                        cols = @intCast(k + 1);
                        break;
                    }
                }
                var rows: u32 = @intCast(slices.len / cols);
                if (rows * cols != slices.len) {
                    rows = 1;
                    cols = @intCast(slices.len);
                }
                if (take.* >= grids.len) return error.MissingCropGrid;
                first_slice_grid = grids[take.*];
                take.* += 1;
                slice_h = first_slice_grid[0] / merge;
                slice_w = first_slice_grid[1] / merge;
                canvas_h = rows * slice_h;
                canvas_w = cols * slice_w;
            }

            fillAll(pos, frame_start, frame_end, origin);
            pos[1][frame_start] = @intCast(halo);
            pos[2][frame_start] = @intCast(halo);
            if (frame.thumb.end < frame_end) {
                pos[1][frame.thumb.end] = @intCast(origin + canvas_h);
                pos[2][frame.thumb.end] = @intCast(origin + canvas_w);
            }
            visionPositions(pos, frame.thumb.start, origin, thumb_grid, merge, canvas_h, canvas_w, 0, 0);

            for (slices, 0..) |sl, k| {
                const h_off: u32 = @intCast((k / cols) * slice_h);
                const w_off: u32 = @intCast((k % cols) * slice_w);
                var grid = first_slice_grid;
                if (k > 0) {
                    if (take.* >= grids.len) return error.MissingCropGrid;
                    grid = grids[take.*];
                    take.* += 1;
                }
                const sh = grid[0] / merge;
                const sw = grid[1] / merge;
                const open = sl.start - 1;
                if (open >= frame_start) {
                    pos[1][open] = @intCast(origin + h_off);
                    pos[2][open] = @intCast(origin + w_off);
                }
                if (sl.end < frame_end) {
                    pos[1][sl.end] = @intCast(origin + h_off + sh - 1);
                    pos[2][sl.end] = @intCast(origin + w_off + sw - 1);
                }
                visionPositions(pos, sl.start, origin, grid, merge, 0, 0, h_off, w_off);
            }

            // The newline ending a row of slices sits just past that row's right edge.
            for (0..slices.len -| 1) |k| {
                const gap_start = slices[k].end;
                const gap_end = slices[k + 1].start;
                if (gap_end - gap_start <= 2) continue;
                const boundary_h = (k / cols + 1) * slice_h - 1;
                const right_edge = cols * slice_w;
                for (gap_start + 1..gap_end - 1) |at| {
                    pos[1][at] = @intCast(origin + boundary_h);
                    pos[2][at] = @intCast(origin + right_edge);
                }
            }

            current_pos = origin + @max(canvas_h, canvas_w) + 1;
            cursor = frame_end;
        }
        if (cursor < group.end) {
            fillRange(pos, cursor, group.end, current_pos);
            current_pos += group.end - cursor;
        }
        current_idx = group.end;
    }
    if (current_idx < n) fillRange(pos, current_idx, n, current_pos);

    var top: i32 = 0;
    for (flat) |v| top = @max(top, v);
    return .{ .pos = flat, .delta = top + 1 - @as(i32, @intCast(n)) };
}

const testing = std.testing;

// `src/fixtures/minicpmv4_7_canvas_rope.json` is `get_rope_index` of transformers' MiniCPMV4_7Model
// on processor-built prompts (tests/dump_minicpmv4_7_vision_fixtures.py --rope-json).
test "canvas positions equal the reference get_rope_index" {
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/minicpmv4_7_canvas_rope.json"), .{});
    defer parsed.deinit();
    const doc = parsed.value.object;
    const u32Of = struct {
        fn get(v: std.json.Value) u32 {
            return @intCast(v.integer);
        }
    }.get;
    const markers = Markers{
        .slice_start = u32Of(doc.get("slice_start_id").?),
        .slice_end = u32Of(doc.get("slice_end_id").?),
        .image_start = u32Of(doc.get("image_start_id").?),
        .image_end = u32Of(doc.get("image_end_id").?),
        .newline = u32Of(doc.get("newline_id").?),
    };
    const cases = doc.get("cases").?.array.items;
    try testing.expect(cases.len >= 6);
    for (cases) |case| {
        const c = case.object;
        const ids_json = c.get("ids").?.array.items;
        const ids = try a.alloc(u32, ids_json.len);
        defer a.free(ids);
        for (ids_json, ids) |v, *id| id.* = u32Of(v);
        const grids_json = c.get("grids").?.array.items;
        const grids = try a.alloc(Grid, grids_json.len);
        defer a.free(grids);
        for (grids_json, grids) |v, *g| g.* = .{ u32Of(v.array.items[0]), u32Of(v.array.items[1]) };

        const got = try canvasPositions(a, ids, .{
            .markers = markers,
            .image_token = u32Of(doc.get("image_token").?),
            .video_token = u32Of(doc.get("video_token").?),
            .merge = u32Of(c.get("merge").?),
        }, grids, &.{});
        defer a.free(got.pos);
        const pos = c.get("pos").?.array.items;
        for (0..3) |row| for (pos[row].array.items, 0..) |want, k| {
            testing.expectEqual(@as(i32, @intCast(want.integer)), got.pos[row * ids.len + k]) catch |e| {
                std.debug.print("{s}: row {d} token {d}\n", .{ c.get("name").?.string, row, k });
                return e;
            };
        };
        try testing.expectEqual(@as(i32, @intCast(c.get("delta").?.integer)), got.delta);
    }
}

test "canvas positions: text-only prompt is plain 1-D" {
    const a = testing.allocator;
    const ids = [_]u32{ 1, 2, 3, 4 };
    const got = try canvasPositions(a, &ids, .{ .markers = .{ .slice_start = 90, .slice_end = 91, .image_start = 92, .image_end = 93, .newline = 94 }, .image_token = 50, .video_token = 51, .merge = 4 }, &.{}, &.{});
    defer a.free(got.pos);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 }, got.pos);
    try testing.expectEqual(@as(i32, 0), got.delta);
}

test "canvas coordinates round ties to even like torch.round" {
    // linspace(0, 3, 7) = 0, .5, 1, 1.5, 2, 2.5, 3
    const want = [_]i32{ 0, 0, 1, 2, 2, 2, 3 };
    for (want, 0..) |w, i| try testing.expectEqual(w, canvasCoord(4, 7, @intCast(i)));
}
