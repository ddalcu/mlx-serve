//! MiniMax H3 vision presentation — the pure math between a reference pixel
//! buffer and the Qwen3-VL sequence it becomes.
//!
//! H3 does NOT condition on a bare prompt when references are present. The
//! reference (`comfy/text_encoders/minimax.py`) splices vision blocks into the
//! raw token stream:
//!
//!     fl2va:  "<Picture 1>: " <|vision_start|> …pad… <|vision_end|> <prompt>
//!     ref2va: per condition, 1-based ordinals PER TYPE, in a fixed order
//!             image -> "<Picture i>: " + block
//!             audio -> "<Audio j>: "            (audio never enters Qwen)
//!             video -> "<Video k>: " then per 2-frame pair
//!                      "<T.T seconds>" + block
//!
//! Four things here decide whether the conditioning lands on the right rows,
//! and every one of them is silent when wrong — the model runs and conditions
//! on garbage rather than erroring. All four are pinned against the reference's
//! OWN output in `fixtures/minimax_h3_vision.json`
//! (`tests/dump_minimax_h3_vision_fixtures.py`):
//!
//!   1. The resize policy. A reference is snapped to a multiple of
//!      patch*merge = 32 with Python's HALF-TO-EVEN rounding, then pushed under
//!      a pixel cap / over a pixel floor. It decides the token count, so an
//!      off-by-one grid shifts every later position.
//!   2. The adaLN modality tags. A vision span carries tag 0 (video) and WIDENS
//!      BY ONE ON EACH SIDE to cover the flanking vision_start/vision_end,
//!      while the mRoPE span does NOT — the two spans are deliberately
//!      different and sharing one would be wrong in both directions.
//!   3. mRoPE position ids. With a vision block present the LM's positions stop
//!      being an arange: the block collapses to ONE t position, its h/w run
//!      over the merged grid, and the text after it resumes at a shifted origin.
//!   4. Qwen3-VL's INTERLEAVED mRoPE: T frequencies by default, H and W
//!      replacing every 3rd slot below `rope_dims[axis]*3`. A section-wise
//!      (contiguous) split is the other common convention and produces a model
//!      that runs and attends wrongly.

const std = @import("std");

// ── Constants (asserted against the reference in the fixture test) ───────────

pub const VISION_START: i32 = 151652;
pub const VISION_END: i32 = 151653;

pub const PATCH: u32 = 16;
pub const TEMPORAL_PATCH: u32 = 2;
pub const MERGE: u32 = 2;
pub const MIN_PIXELS: u32 = 3136;
pub const MAX_PIXELS: u32 = 12845056;

/// Qwen3-VL normalizes to [-1, 1] (mean/std 0.5), not CLIP statistics.
pub const IMAGE_MEAN: f32 = 0.5;
pub const IMAGE_STD: f32 = 0.5;

/// LM rope: theta and the per-axis mRoPE section widths (T, H, W).
pub const ROPE_THETA: f64 = 5_000_000.0;
pub const ROPE_DIMS = [3]u32{ 24, 20, 20 };

/// Qwen3-VL-32B's vision tower geometry, from the reference's
/// `QWEN3VL_VISION["qwen3vl_32b"]` + `QWEN3VL_VISION_COMMON`.
pub const VIT_HIDDEN: c_int = 1152;
pub const VIT_HEADS: c_int = 16;
pub const VIT_HEAD_DIM: c_int = 72;
pub const VIT_INTER: c_int = 4304;
pub const VIT_DEPTH: usize = 27;
pub const VIT_OUT: c_int = 5120;
pub const VIT_DEEPSTACK = [3]usize{ 8, 16, 24 };
pub const VIT_GRID_SIDE: usize = 48; // sqrt(num_position_embeddings 2304)

/// Python's `round`, i.e. half-to-EVEN. `height / 32` lands exactly on .5 for
/// every height ≡ 16 (mod 32) — 3024 is one, and it is a real phone-camera
/// dimension — where half-away-from-zero picks the other 32-pixel step.
pub fn roundHalfEven(x: f64) f64 {
    const fl = @floor(x);
    const frac = x - fl;
    if (frac > 0.5) return fl + 1;
    if (frac < 0.5) return fl;
    return if (@mod(fl, 2.0) == 0.0) fl else fl + 1;
}

// ── Resize policy ───────────────────────────────────────────────────────────

pub const Canvas = struct { h: u32, w: u32 };

/// Qwen3-VL's `process_qwen2vl_images` sizing: snap each axis to
/// patch*merge = 32, then scale under the pixel cap / over the pixel floor.
/// Note the cap and floor branches scale from the ORIGINAL dimensions, not the
/// snapped ones, and use floor/ceil rather than the half-even round.
pub fn fitCanvas(h: u32, w: u32) Canvas {
    const factor: f64 = @floatFromInt(PATCH * MERGE);
    const fh: f64 = @floatFromInt(h);
    const fw: f64 = @floatFromInt(w);
    var h_bar = roundHalfEven(fh / factor) * factor;
    var w_bar = roundHalfEven(fw / factor) * factor;

    const max_px: f64 = @floatFromInt(MAX_PIXELS);
    const min_px: f64 = @floatFromInt(MIN_PIXELS);
    if (h_bar * w_bar > max_px) {
        const beta = @sqrt((fh * fw) / max_px);
        h_bar = @max(factor, @floor(fh / beta / factor) * factor);
        w_bar = @max(factor, @floor(fw / beta / factor) * factor);
    } else if (h_bar * w_bar < min_px) {
        const beta = @sqrt(min_px / (fh * fw));
        h_bar = @ceil(fh * beta / factor) * factor;
        w_bar = @ceil(fw * beta / factor) * factor;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

/// A vision block's patch grid. `t` is the TEMPORAL PATCH count, always 1 here:
/// a still image repeats itself to fill the 2-frame patch and a video block is
/// exactly one 2-frame pair, so both produce grid_t = 1.
pub const Grid = struct {
    t: u32,
    gh: u32,
    gw: u32,

    /// Raw patches, i.e. rows entering the ViT.
    pub fn patches(self: Grid) u32 {
        return self.t * self.gh * self.gw;
    }

    /// Rows entering the LM after the merger's 2x2 spatial shuffle — this is
    /// the `size` of the span, not `patches`.
    pub fn mergedTokens(self: Grid) u32 {
        return self.patches() / (MERGE * MERGE);
    }

    pub fn mergedH(self: Grid) u32 {
        return self.gh / MERGE;
    }
    pub fn mergedW(self: Grid) u32 {
        return self.gw / MERGE;
    }
};

/// The grid a reference of `h`x`w` pixels becomes. Identical for a still image
/// and a 2-frame video pair — `process_video_block` reuses the image policy and
/// only changes what fills the temporal patch.
pub fn gridFor(h: u32, w: u32) Grid {
    const c = fitCanvas(h, w);
    return .{ .t = 1, .gh = c.h / PATCH, .gw = c.w / PATCH };
}

// ── Spans, tags and positions ───────────────────────────────────────────────

/// One vision block in the LM sequence. `index` is the FIRST expanded row, i.e.
/// one past its `<|vision_start|>`; `size` is `grid.mergedTokens()`.
pub const Span = struct {
    index: u32,
    size: u32,
    grid: Grid,

    pub fn end(self: Span) u32 {
        return self.index + self.size;
    }
};

/// AdaLN modality tag per LM position: 1 = text, 0 = video (the vision pads).
/// The vision run widens by ONE on each side so the flanking vision_start /
/// vision_end tokens ride the video modality too. Caller owns the slice.
pub fn tokenTags(allocator: std.mem.Allocator, seq_len: u32, spans: []const Span) ![]u8 {
    const out = try allocator.alloc(u8, seq_len);
    errdefer allocator.free(out);
    @memset(out, 1);
    for (spans) |sp| {
        const lo = if (sp.index == 0) 0 else sp.index - 1;
        const hi = @min(seq_len, sp.end() + 1);
        for (out[lo..hi]) |*t| t.* = 0;
    }
    return out;
}

pub const TagRun = struct { start: u32, end: u32, tag: u8 };

/// Maximal runs of equal tag. The DiT consumes runs (one adaLN modulation row
/// each), never a per-position tag vector. Caller owns the slice.
pub fn tagRuns(allocator: std.mem.Allocator, tags: []const u8) ![]TagRun {
    var list: std.ArrayList(TagRun) = .empty;
    errdefer list.deinit(allocator);
    if (tags.len == 0) return list.toOwnedSlice(allocator);
    var run_start: u32 = 0;
    var i: u32 = 1;
    while (i <= tags.len) : (i += 1) {
        if (i == tags.len or tags[i] != tags[run_start]) {
            try list.append(allocator, .{ .start = run_start, .end = i, .tag = tags[run_start] });
            run_start = i;
        }
    }
    return list.toOwnedSlice(allocator);
}

/// [3 * seq_len] mRoPE position ids, AXIS-MAJOR (`out[axis * seq_len + i]`), or
/// null when there is no vision block — which is the signal to use plain 1-D
/// rope, not a 3-row arange (the reference branches on the row count).
///
/// Ported from `qwen_vl.qwen2vl_mrope_position_ids`. Three things are load-
/// bearing: the whole block collapses to ONE t position; h/w run row-major over
/// the MERGED grid; and the text after a block resumes at `max(grid)/2` past
/// the block's start, not past its end — which is why `offset` accumulates
/// `len_max - size` and every later position shifts.
pub fn mropePositions(allocator: std.mem.Allocator, seq_len: u32, spans: []const Span) !?[]f64 {
    if (spans.len == 0) return null;
    const out = try allocator.alloc(f64, @as(usize, seq_len) * 3);
    errdefer allocator.free(out);
    @memset(out, 0);

    var offset: i64 = 0;
    for (spans, 0..) |sp, si| {
        const start = sp.index;
        const end = sp.end();
        if (si == 0) {
            for (0..start) |i| {
                const v: f64 = @floatFromInt(i);
                for (0..3) |ax| out[ax * seq_len + i] = v;
            }
        }
        const len_max: i64 = @intCast(@max(sp.grid.t, @max(sp.grid.gh, sp.grid.gw)) / 2);
        const start_next: i64 = len_max + @as(i64, start);
        var k: u32 = end;
        while (k < seq_len) : (k += 1) {
            const v: f64 = @floatFromInt(start_next + offset + @as(i64, k - end));
            for (0..3) |ax| out[ax * seq_len + k] = v;
        }

        const base: f64 = @floatFromInt(@as(i64, start) + offset);
        const size = sp.size;
        // t: one position for the whole block.
        for (start..end) |i| out[0 * seq_len + i] = base;
        // h: `ceil(size / merged_h)` consecutive repeats of each row index.
        const mh = sp.grid.mergedH();
        if (mh > 0) {
            const rep = std.math.divCeil(u32, size, mh) catch 1;
            for (0..size) |i| out[1 * seq_len + start + i] = base + @as(f64, @floatFromInt(i / rep));
        }
        // w: the column indices cycling.
        const mw = sp.grid.mergedW();
        if (mw > 0) {
            for (0..size) |i| out[2 * seq_len + start + i] = base + @as(f64, @floatFromInt(i % mw));
        }
        offset += len_max - @as(i64, size);
    }
    return out;
}

// ── Interleaved mRoPE ───────────────────────────────────────────────────────

/// Which position axis feeds frequency slot `j` of `head_dim / 2`.
///
/// Qwen3-VL interleaves rather than sectioning: T is the default everywhere, H
/// takes slots ≡ 1 (mod 3) and W slots ≡ 2 (mod 3), both only below
/// `rope_dims[axis] * 3`. The tail past `3 * min(...)` therefore stays T, which
/// is what makes the counts come out (24, 20, 20) rather than (22, 21, 21).
pub fn axisOfFreq(j: usize) u2 {
    const m = j % 3;
    if (m == 1 and j < ROPE_DIMS[1] * 3) return 1;
    if (m == 2 and j < ROPE_DIMS[2] * 3) return 2;
    return 0;
}

/// Per-position rope ANGLES [seq_len * (head_dim/2)].
///
/// `positions` is the axis-major [3 * seq_len] table from `mropePositions`, or
/// null for the text-only case where every axis shares the running index and
/// the interleave collapses to plain 1-D rope. Caller owns the slice.
pub fn ropeAngles(
    allocator: std.mem.Allocator,
    seq_len: u32,
    positions: ?[]const f64,
    head_dim: usize,
    theta: f64,
) ![]f64 {
    const half = head_dim / 2;
    const out = try allocator.alloc(f64, @as(usize, seq_len) * half);
    errdefer allocator.free(out);
    for (0..half) |j| {
        const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(head_dim)));
        const ax: usize = if (positions == null) 0 else axisOfFreq(j);
        for (0..seq_len) |i| {
            const p: f64 = if (positions) |pp| pp[ax * seq_len + i] else @floatFromInt(i);
            out[i * half + j] = p * inv;
        }
    }
    return out;
}

// ── Presentation text ───────────────────────────────────────────────────────

/// The label that precedes a reference block. Ordinals are 1-based PER TYPE, so
/// a prompt can address them as `<Picture 1>` / `<Video 1>` / `<Audio 1>` — the
/// wording is a contract with the checkpoint, not a cosmetic choice.
pub const RefKind = enum { image, audio, video };

pub fn labelFor(buf: []u8, kind: RefKind, ordinal: u32) ![]const u8 {
    return switch (kind) {
        .image => std.fmt.bufPrint(buf, "<Picture {d}>: ", .{ordinal}),
        .audio => std.fmt.bufPrint(buf, "<Audio {d}>: ", .{ordinal}),
        .video => std.fmt.bufPrint(buf, "<Video {d}>: ", .{ordinal}),
    };
}

/// The timestamp label before each 2-frame video block, at the pair's MIDPOINT.
/// One decimal, matching `"<%.1f seconds>"`.
pub fn timestampLabel(buf: []u8, seconds: f64) ![]const u8 {
    return std.fmt.bufPrint(buf, "<{d:.1} seconds>", .{seconds});
}

// ── Tests ───────────────────────────────────────────────────────────────────

const testing = std.testing;
const vision_fixture = @embedFile("fixtures/minimax_h3_vision.json");

fn loadVisionFixture(allocator: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, vision_fixture, .{});
}

fn jf(v: std.json.Value) f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => std.math.nan(f64),
    };
}

fn ju(v: std.json.Value) u32 {
    return switch (v) {
        .integer => |i| @intCast(i),
        else => std.math.maxInt(u32),
    };
}

test "minimax h3 vision: constants match the reference" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();
    const c = parsed.value.object.get("constants").?.object;

    try testing.expectEqual(@as(u32, @intCast(VISION_START)), ju(c.get("VISION_START").?));
    try testing.expectEqual(@as(u32, @intCast(VISION_END)), ju(c.get("VISION_END").?));
    try testing.expectEqual(PATCH, ju(c.get("patch_size").?));
    try testing.expectEqual(TEMPORAL_PATCH, ju(c.get("temporal_patch_size").?));
    try testing.expectEqual(MERGE, ju(c.get("merge_size").?));
    try testing.expectEqual(MIN_PIXELS, ju(c.get("min_pixels").?));
    try testing.expectEqual(MAX_PIXELS, ju(c.get("max_pixels").?));
    try testing.expectApproxEqAbs(ROPE_THETA, jf(c.get("rope_theta").?), 1e-6);
    for (c.get("rope_dims").?.array.items, 0..) |v, i|
        try testing.expectEqual(ROPE_DIMS[i], ju(v));
    for (c.get("QWEN_IMAGE_MEAN").?.array.items) |v|
        try testing.expectApproxEqAbs(@as(f64, IMAGE_MEAN), jf(v), 1e-9);
    for (c.get("QWEN_IMAGE_STD").?.array.items) |v|
        try testing.expectApproxEqAbs(@as(f64, IMAGE_STD), jf(v), 1e-9);

    // Tower geometry: a wrong depth or deepstack index loads a checkpoint that
    // is present-and-silent rather than missing-and-loud.
    try testing.expectEqual(@as(u32, @intCast(VIT_HIDDEN)), ju(c.get("vit_hidden").?));
    try testing.expectEqual(@as(u32, VIT_DEPTH), ju(c.get("vit_depth").?));
    try testing.expectEqual(@as(u32, @intCast(VIT_INTER)), ju(c.get("vit_intermediate").?));
    try testing.expectEqual(@as(u32, @intCast(VIT_HEADS)), ju(c.get("vit_heads").?));
    try testing.expectEqual(@as(u32, @intCast(VIT_OUT)), ju(c.get("vit_out_hidden").?));
    try testing.expectEqual(@as(u32, @intCast(VIT_HIDDEN)) * @as(u32, @intCast(VIT_HEADS)) / @as(u32, @intCast(VIT_HEADS)), ju(c.get("vit_hidden").?));
    try testing.expectEqual(@as(c_int, @divExact(VIT_HIDDEN, VIT_HEADS)), VIT_HEAD_DIM);
    for (c.get("vit_deepstack_indexes").?.array.items, 0..) |v, i|
        try testing.expectEqual(@as(u32, @intCast(VIT_DEEPSTACK[i])), ju(v));
    try testing.expectEqual(@as(u32, @intCast(VIT_GRID_SIDE * VIT_GRID_SIDE)), ju(c.get("vit_num_position_embeddings").?));
}

test "minimax h3 vision: resize policy matches the reference grids" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();

    for (parsed.value.object.get("image_grids").?.array.items) |item| {
        const o = item.object;
        const label = o.get("label").?.string;
        const g = gridFor(ju(o.get("in_h").?), ju(o.get("in_w").?));
        const want = o.get("grid_thw").?.array.items;
        testing.expectEqual(ju(want[0]), g.t) catch |e| {
            std.debug.print("case {s}: grid_t\n", .{label});
            return e;
        };
        testing.expectEqual(ju(want[1]), g.gh) catch |e| {
            std.debug.print("case {s}: grid_h (got {d})\n", .{ label, g.gh });
            return e;
        };
        testing.expectEqual(ju(want[2]), g.gw) catch |e| {
            std.debug.print("case {s}: grid_w (got {d})\n", .{ label, g.gw });
            return e;
        };
        try testing.expectEqual(ju(o.get("n_patches").?), g.patches());
        try testing.expectEqual(ju(o.get("merged_tokens").?), g.mergedTokens());
        // The ViT patch row width is fixed by the policy, not by the image.
        try testing.expectEqual(@as(u32, 3 * TEMPORAL_PATCH * PATCH * PATCH), ju(o.get("patch_dim").?));
    }

    // A 2-frame video pair reuses the image policy verbatim: only what fills
    // the temporal patch differs, never the grid.
    for (parsed.value.object.get("video_blocks").?.array.items) |item| {
        const o = item.object;
        const g = gridFor(ju(o.get("in_h").?), ju(o.get("in_w").?));
        const want = o.get("grid_thw").?.array.items;
        try testing.expectEqual(ju(want[1]), g.gh);
        try testing.expectEqual(ju(want[2]), g.gw);
        try testing.expectEqual(ju(o.get("merged_tokens").?), g.mergedTokens());
    }
}

fn spansFromFixture(a: std.mem.Allocator, o: std.json.ObjectMap) ![]Span {
    const raw = o.get("spans").?.array.items;
    const out = try a.alloc(Span, raw.len);
    for (raw, 0..) |sp, i| {
        const arr = sp.array.items;
        const g = arr[2].array.items;
        out[i] = .{
            .index = ju(arr[0]),
            .size = ju(arr[1]),
            .grid = .{ .t = ju(g[0]), .gh = ju(g[1]), .gw = ju(g[2]) },
        };
    }
    return out;
}

test "minimax h3 vision: token tags widen over the vision delimiters" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();

    for (parsed.value.object.get("token_tags").?.array.items) |item| {
        const o = item.object;
        const label = o.get("label").?.string;
        const seq = ju(o.get("seq_len").?);
        const spans = try spansFromFixture(a, o);
        defer a.free(spans);

        const tags = try tokenTags(a, seq, spans);
        defer a.free(tags);
        const runs = try tagRuns(a, tags);
        defer a.free(runs);

        const want = o.get("runs").?.array.items;
        testing.expectEqual(want.len, runs.len) catch |e| {
            std.debug.print("case {s}: run count {d} vs {d}\n", .{ label, runs.len, want.len });
            return e;
        };
        for (want, runs) |wv, got| {
            const w = wv.array.items;
            testing.expectEqual(ju(w[0]), got.start) catch |e| {
                std.debug.print("case {s}: run start\n", .{label});
                return e;
            };
            try testing.expectEqual(ju(w[1]), got.end);
            try testing.expectEqual(@as(u8, @intCast(ju(w[2]))), got.tag);
        }
        var sum: u32 = 0;
        for (tags) |t| sum += t;
        try testing.expectEqual(ju(o.get("tag_sum").?), sum);
    }
}

test "minimax h3 vision: mRoPE position ids match the reference" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();

    for (parsed.value.object.get("mrope_position_ids").?.array.items) |item| {
        const o = item.object;
        const label = o.get("label").?.string;
        const seq = ju(o.get("seq_len").?);
        const present = o.get("present").?.bool;
        if (!present) {
            const none = try mropePositions(a, seq, &.{});
            try testing.expect(none == null);
            continue;
        }
        const spans = try spansFromFixture(a, o);
        defer a.free(spans);
        const pos = (try mropePositions(a, seq, spans)).?;
        defer a.free(pos);

        for (o.get("rows").?.array.items, 0..) |rowv, ax| {
            for (rowv.array.items, 0..) |v, i| {
                testing.expectApproxEqAbs(jf(v), pos[ax * seq + i], 1e-9) catch |e| {
                    std.debug.print("case {s}: axis {d} pos {d}: want {d} got {d}\n", .{ label, ax, i, jf(v), pos[ax * seq + i] });
                    return e;
                };
            }
        }
        // A plain column sum is permutation-invariant; the row-weighted one is
        // what a reordered span cannot survive.
        for (0..3) |ax| {
            var sum: f64 = 0;
            var wsum: f64 = 0;
            for (0..seq) |i| {
                sum += pos[ax * seq + i];
                wsum += @as(f64, @floatFromInt(i + 1)) * pos[ax * seq + i];
            }
            try testing.expectApproxEqRel(jf(o.get("row_sums").?.array.items[ax]), sum, 1e-9);
            try testing.expectApproxEqRel(jf(o.get("row_weighted").?.array.items[ax]), wsum, 1e-9);
        }
    }
}

test "minimax h3 vision: interleaved mRoPE picks the reference's axis per slot" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();

    const m = parsed.value.object.get("interleave_map").?.object;
    const want = m.get("axis_of_freq").?.array.items;
    for (want, 0..) |v, j| {
        testing.expectEqual(@as(u2, @intCast(ju(v))), axisOfFreq(j)) catch |e| {
            std.debug.print("freq slot {d}: want axis {d} got {d}\n", .{ j, ju(v), axisOfFreq(j) });
            return e;
        };
    }
    var counts = [3]u32{ 0, 0, 0 };
    for (0..want.len) |j| counts[axisOfFreq(j)] += 1;
    for (m.get("counts").?.array.items, 0..) |v, i|
        try testing.expectEqual(ju(v), counts[i]);
}

test "minimax h3 vision: rope angles reproduce the reference cos/sin" {
    const a = testing.allocator;
    var parsed = try loadVisionFixture(a);
    defer parsed.deinit();

    for (parsed.value.object.get("interleaved_rope").?.array.items) |item| {
        const o = item.object;
        const label = o.get("label").?.string;
        const seq = ju(o.get("seq_len").?);
        const head_dim: usize = @intCast(ju(o.get("head_dim").?));
        const half = head_dim / 2;

        var spans: []Span = &.{};
        if (o.get("spans")) |_| spans = try spansFromFixture(a, o);
        defer if (spans.len > 0) a.free(spans);
        const pos = try mropePositions(a, seq, spans);
        defer if (pos) |p| a.free(p);

        const ang = try ropeAngles(a, seq, pos, head_dim, ROPE_THETA);
        defer a.free(ang);

        // The reference's emb is cat(freqs, freqs), so cos is the angle cosine
        // duplicated across the two halves.
        //
        // Tolerance 1e-5, not 1e-9: the reference forms the angle with an f32
        // matmul (`inv_freq.float() @ position_ids.float()`) while we
        // accumulate in f64, so at position ~35 the two disagree at f32's own
        // resolution (~2e-6) — and both sides then round to bf16, which resolves
        // ~1e-3. This cannot hide the failure the test exists for: a slot taking
        // the WRONG position axis moves cos by O(1), and the axis map is pinned
        // exactly by the interleave_map test above.
        const cf = o.get("cos_first").?.array.items;
        for (0..head_dim) |j| {
            const want = jf(cf[j]);
            const got = @cos(ang[0 * half + (j % half)]);
            testing.expectApproxEqAbs(want, got, 1e-5) catch |e| {
                std.debug.print("case {s}: cos_first[{d}] want {d} got {d}\n", .{ label, j, want, got });
                return e;
            };
        }
        const cl = o.get("cos_last").?.array.items;
        for (0..head_dim) |j|
            try testing.expectApproxEqAbs(jf(cl[j]), @cos(ang[(seq - 1) * half + (j % half)]), 1e-5);

        var cs: f64 = 0;
        var ss: f64 = 0;
        var cw: f64 = 0;
        var sw: f64 = 0;
        for (0..seq) |i| {
            var rc: f64 = 0;
            var rs: f64 = 0;
            for (0..half) |j| {
                rc += @cos(ang[i * half + j]);
                rs += @sin(ang[i * half + j]);
            }
            // cos is duplicated over both halves; sin_lo is only the low half.
            cs += 2 * rc;
            ss += rs;
            cw += @as(f64, @floatFromInt(i + 1)) * 2 * rc;
            sw += @as(f64, @floatFromInt(i + 1)) * rs;
        }
        testing.expectApproxEqRel(jf(o.get("cos_sum").?), cs, 1e-4) catch |e| {
            std.debug.print("case {s}: cos_sum want {d} got {d}\n", .{ label, jf(o.get("cos_sum").?), cs });
            return e;
        };
        try testing.expectApproxEqRel(jf(o.get("sin_sum").?), ss, 1e-4);
        try testing.expectApproxEqRel(jf(o.get("cos_weighted").?), cw, 1e-4);
        try testing.expectApproxEqRel(jf(o.get("sin_weighted").?), sw, 1e-4);
    }
}

test "minimax h3 vision: presentation labels are the reference's wording" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("<Picture 1>: ", try labelFor(&buf, .image, 1));
    try testing.expectEqualStrings("<Audio 2>: ", try labelFor(&buf, .audio, 2));
    try testing.expectEqualStrings("<Video 3>: ", try labelFor(&buf, .video, 3));
    try testing.expectEqualStrings("<0.0 seconds>", try timestampLabel(&buf, 0.0));
    try testing.expectEqualStrings("<12.5 seconds>", try timestampLabel(&buf, 12.5));
    // One decimal, not the shortest representation: "<1.0 seconds>" is what the
    // checkpoint saw, and "<1 seconds>" is a different token sequence.
    try testing.expectEqualStrings("<1.0 seconds>", try timestampLabel(&buf, 1.0));
}
