//! What a media plugin's `step` needs around its backend: the table's `Progress` as the backend's `progress.Progress`,
//! a request body's top-level fields, and a job error as a `Verdict`.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const mlx = @import("../../mlx/ffi.zig");
const progress = @import("progress.zig");
const webp = @import("webp");
const png = @import("png.zig");

const Verdict = abi.Verdict;

/// `p` as the progress the backends take: `boundary` answers both the per-step poll and the cancel probe, and the
/// table's `boundary` is where the host gives chat its turn (once per step, never also here).
pub fn progressOf(p: *const abi.Progress) progress.Progress {
    const o = p.preview_opts;
    return .{
        .ctx = @constCast(p),
        .cb = report,
        .cancelled_cb = boundary,
        .yields = false,
        .preview_cb = if (p.preview != null and o.enabled) preview else null,
        .preview_opts = .{ .enabled = o.enabled, .frames = o.frames, .max_side = o.max_side },
    };
}

fn preview(ctx: *anyopaque, stage: []const u8, step: u32, total: u32, frame: @import("preview.zig").Encoded) void {
    const p: *const abi.Progress = @ptrCast(@alignCast(ctx));
    const f: abi.PreviewFrame = .{ .bytes = .of(frame.jpeg), .mime = .of(frame.mime), .w = frame.w, .h = frame.h };
    p.preview.?(p.ctx, .of(stage), step, total, &f);
}

fn report(ctx: *anyopaque, stage: []const u8, step: u32, total: u32) void {
    const p: *const abi.Progress = @ptrCast(@alignCast(ctx));
    p.report(p.ctx, .of(stage), step, total);
}

fn boundary(ctx: *anyopaque) bool {
    const p: *const abi.Progress = @ptrCast(@alignCast(ctx));
    return p.boundary(p.ctx);
}

/// A request's JSON object. `parse` refuses anything but an object.
pub const Body = struct {
    parsed: std.json.Parsed(std.json.Value),

    pub fn parse(a: std.mem.Allocator, json: []const u8) ?Body {
        const p = std.json.parseFromSlice(std.json.Value, a, json, .{}) catch return null;
        if (p.value != .object) {
            p.deinit();
            return null;
        }
        return .{ .parsed = p };
    }

    pub fn deinit(b: *Body) void {
        b.parsed.deinit();
    }

    pub fn get(b: *const Body, key: []const u8) ?std.json.Value {
        return b.parsed.value.object.get(key);
    }

    pub fn str(b: *const Body, key: []const u8) ?[]const u8 {
        const v = b.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    /// True only for the literal `true`.
    pub fn flag(b: *const Body, key: []const u8) bool {
        const v = b.get(key) orelse return false;
        return v == .bool and v.bool;
    }

    pub fn float(b: *const Body, key: []const u8) ?f64 {
        const v = b.get(key) orelse return null;
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => null,
        };
    }

    /// A non-negative integer field; a negative, fractional or non-number value reads as absent.
    pub fn uint(b: *const Body, key: []const u8) ?u64 {
        const v = b.get(key) orelse return null;
        return if (v == .integer and v.integer >= 0) @intCast(v.integer) else null;
    }

    /// A whole non-negative number: null when absent, `error.NotWhole` for any other value (4.9, -1, "4"), which the
    /// caller refuses by name.
    pub fn whole(b: *const Body, key: []const u8) error{NotWhole}!?u64 {
        const v = b.get(key) orelse return null;
        return if (v == .integer and v.integer >= 0) @intCast(v.integer) else error.NotWhole;
    }
};

/// What the GPU can still take: Metal's working set less live MLX memory.
pub fn gpuHeadroom() u64 {
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    return mlx.maxRecommendedWorkingSet() -| @as(u64, active);
}

/// A failed job: `busy` when it ran out of memory (the machine's capacity, as chat's 503), else `failed`.
pub fn failure(err: anyerror, comptime what: []const u8) Verdict {
    const oom = err == error.OutOfMemory or mlx.gpuContextLost() or
        if (mlx.peekErrorName()) |n| std.mem.eql(u8, n, "OutOfMemory") else false;
    return Verdict.refuse(if (oom) .busy else .failed, what);
}

pub const client_left = Verdict.refuse(.failed, "the client left");

/// A failure the GPU latched during the job, by class, else null. A stage the job reads on the host comes back as
/// zeros when its command buffer failed: without this a 3D job answered 200 with a zeroed GLB.
pub fn latched() ?Verdict {
    return if (mlx.peekErrorName() != null) failure(error.MlxFailure, "generation failed") else null;
}

/// `instrumental: true` beside words to sing is contradictory. Letting either
/// side quietly win is the failure mode — a sticky checkbox silently discards a
/// verse the user typed, or the checkbox does nothing — so the pair is a NAMED
/// 400 (the `/v1/images/edits` rule: everything we cannot honor is named).
/// Whitespace-only lyrics count as ABSENT so an app that keeps a blank lyrics
/// editor mounted beside the checkbox is fine. Both backends read this ONE
/// predicate, so the rule cannot drift between them.
pub fn instrumentalConflicts(instrumental: bool, lyrics: []const u8) bool {
    return instrumental and std.mem.trim(u8, lyrics, " \t\r\n").len != 0;
}

/// Standard base64 into an owned buffer (media fields ride JSON bodies as base64).
pub fn base64Decode(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const out = try a.alloc(u8, try dec.calcSizeForSlice(text));
    errdefer a.free(out);
    try dec.decode(out, text);
    return out;
}

/// An uploaded image (base64) as the media backends read it through stb_image: a WebP re-encoded as a lossless PNG
/// (alpha kept), anything else exactly as sent. Caller frees.
pub fn imageUpload(a: std.mem.Allocator, text: []const u8) ![]u8 {
    const bytes = try base64Decode(a, text);
    var w: c_int = 0;
    var h: c_int = 0;
    const rgba = webp.WebPDecodeRGBA(bytes.ptr, bytes.len, &w, &h) orelse return bytes;
    defer webp.WebPFree(rgba);
    defer a.free(bytes);
    const n = @as(usize, @intCast(w)) * @as(usize, @intCast(h)) * 4;
    return png.encodeRgba(a, rgba[0..n], @intCast(w), @intCast(h));
}

// ── tests ──

const testing = std.testing;

test "plugin job: a body reads typed fields and refuses a non-object" {
    try testing.expect(Body.parse(testing.allocator, "[1]") == null);
    try testing.expect(Body.parse(testing.allocator, "{") == null);
    var b = Body.parse(testing.allocator, "{\"s\":\"x\",\"f\":true,\"n\":-1,\"u\":7,\"r\":0.5}").?;
    defer b.deinit();
    try testing.expectEqualStrings("x", b.str("s").?);
    try testing.expect(b.flag("f") and !b.flag("s"));
    try testing.expect(b.uint("n") == null and b.uint("u").? == 7);
    try testing.expectEqual(@as(f64, 0.5), b.float("r").?);
    try testing.expect(failure(error.Boom, "x").code == .failed and failure(error.OutOfMemory, "x").code == .busy);
}

test "plugin job: whole reads a non-negative integer, refuses any other value, and is null when absent" {
    var b = Body.parse(testing.allocator, "{\"a\":4,\"b\":4.9,\"c\":-1,\"d\":\"4\"}").?;
    defer b.deinit();
    try testing.expectEqual(@as(?u64, 4), try b.whole("a"));
    try testing.expectEqual(@as(?u64, null), try b.whole("e"));
    for ([_][]const u8{ "b", "c", "d" }) |k| try testing.expectError(error.NotWhole, b.whole(k));
}

test "instrumental is a request-level rule: the flag and real lyrics are a conflict, not a race" {
    // `instrumental: true` beside words to sing is contradictory, and letting
    // either side quietly win is the failure mode — a user who typed a verse
    // and left a sticky checkbox set gets a wordless track with no explanation,
    // or a checkbox that does nothing. Both backends read the ONE predicate, so
    // the rule cannot drift between them.
    try std.testing.expect(instrumentalConflicts(true, "[verse]\nhello"));
    try std.testing.expect(instrumentalConflicts(true, "la la la"));
    // Whitespace-only is ABSENT, not a conflict: an app that keeps a blank
    // lyrics editor mounted beside the checkbox must not 400.
    try std.testing.expect(!instrumentalConflicts(true, ""));
    try std.testing.expect(!instrumentalConflicts(true, "  \n\t\r "));
    // The flag off never conflicts, whatever the lyrics say.
    try std.testing.expect(!instrumentalConflicts(false, "[verse]\nhello"));
    try std.testing.expect(!instrumentalConflicts(false, ""));
}
