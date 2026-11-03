//! A plugin's media hooks (`abi.MediaHooks`) as the engine's encoder arm: a decoded item prepared as the model's
//! processor does, through the plugin's tower, and its placeholder run laid out with the model's tokenizer. The run is
//! kept under the item's encoder-cache key, so a cached item still expands.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const mlx = @import("../../../mlx/ffi.zig");
const Tokenizer = @import("../../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const log = @import("../../../sdk/util/log.zig");

pub const PluginMedia = struct {
    hooks: *const abi.MediaHooks,
    model: *anyopaque,
    parsed: *const abi.Parsed,
    a: std.mem.Allocator,
    runs: std.AutoHashMapUnmanaged(u64, []u32) = .{},

    /// Runs kept before the map starts over (a run is a few hundred ids).
    const MAX_RUNS = 512;

    pub fn deinit(pm: *PluginMedia) void {
        pm.clearRuns();
        pm.runs.deinit(pm.a);
    }

    fn clearRuns(pm: *PluginMedia) void {
        var it = pm.runs.valueIterator();
        while (it.next()) |r| pm.a.free(r.*);
        pm.runs.clearRetainingCapacity();
    }

    /// The item's rows `[1, rows, hidden]` (caller owns); its run is kept under `key`.
    pub fn encode(pm: *PluginMedia, in: abi.MediaInput, key: u64, budget: u64, tok: ?*const Tokenizer) !mlx.mlx_array {
        var item: abi.MediaItem = .{};
        try check("prepare", pm.hooks.prepare(pm.parsed, &in, budget, &item));
        defer if (item.pixels.ctx != null) {
            _ = mlx.mlx_array_free(.{ .ctx = item.pixels.ctx });
        };
        var out: abi.Array = .{};
        try check("tower", pm.hooks.tower(pm.model, &item, &out));
        var rows: mlx.mlx_array = .{ .ctx = out.ctx };
        errdefer _ = mlx.mlx_array_free(rows);
        if (rows.ctx == null) return error.PluginMediaFailed;
        const n = mlx.mlx_array_shape(rows)[if (mlx.mlx_array_ndim(rows) == 3) 1 else 0];
        if (mlx.mlx_array_ndim(rows) == 2) {
            var r3 = mlx.mlx_array_new();
            const sh = mlx.mlx_array_shape(rows);
            try mlx.check(mlx.mlx_reshape(&r3, rows, &[_]c_int{ 1, sh[0], sh[1] }, 3, mlx.gpuStream()));
            _ = mlx.mlx_array_free(rows);
            rows = r3;
        }
        try pm.keepRun(key, &item, @intCast(n), tok);
        return rows;
    }

    /// The run kept for `key`, if any.
    pub fn run(pm: *const PluginMedia, key: u64) ?[]const u32 {
        return pm.runs.get(key);
    }

    fn keepRun(pm: *PluginMedia, key: u64, item: *const abi.MediaItem, rows: u32, tok: ?*const Tokenizer) !void {
        const r = try runOf(pm.a, pm.hooks, pm.parsed, item, rows, tok);
        errdefer pm.a.free(r);
        if (pm.runs.count() >= MAX_RUNS) pm.clearRuns();
        const slot = try pm.runs.getOrPut(pm.a, key);
        if (slot.found_existing) pm.a.free(slot.value_ptr.*);
        slot.value_ptr.* = r;
    }
};

fn check(comptime what: []const u8, v: abi.Verdict) !void {
    if (v.isOk()) return;
    log.err("[plugin] media {s} refused: {s}\n", .{ what, v.why.slice() });
    return if (v.code == .unsupported) error.MediaUnsupported else error.PluginMediaFailed;
}

/// The placeholder id of `kind` (the run a plugin without `placeholder_rows` gets, and what the splice fills).
fn placeholderOf(p: *const abi.Parsed, kind: abi.MediaKind) u32 {
    return switch (kind) {
        .image => p.placeholder_ids.image,
        .video => p.placeholder_ids.video,
        .audio => p.placeholder_ids.audio,
    };
}

/// The item's run, owned; it must carry exactly `rows` of its kind's placeholder id (the rows the splice fills).
pub fn runOf(a: std.mem.Allocator, hooks: *const abi.MediaHooks, p: *const abi.Parsed, item: *const abi.MediaItem, rows: u32, tok: ?*const Tokenizer) ![]u32 {
    const id = placeholderOf(p, item.kind);
    if (id == 0) return error.MediaPlaceholderMismatch;
    const write = hooks.placeholder_rows orelse {
        const r = try a.alloc(u32, rows);
        @memset(r, id);
        return r;
    };
    // Room for the rows and any text the plugin writes around them (a video's timestamps).
    const cap = rows * 2 + 4096;
    const buf = try a.alloc(u32, cap);
    errdefer a.free(buf);
    const tt: abi.TextTokens = .{ .ctx = @ptrCast(@constCast(tok)), .encode = &encodeText };
    const n = write(p, item, &tt, buf.ptr, cap);
    if (n > cap) {
        log.err("[plugin] media run of {d} ids past its room of {d}\n", .{ n, cap });
        return error.MediaPlaceholderMismatch;
    }
    var held: u32 = 0;
    for (buf[0..n]) |t| held += @intFromBool(t == id);
    if (n == 0 or held != rows) {
        log.err("[plugin] media run of {d} ids holds {d} placeholders for {d} rows\n", .{ n, held, rows });
        return error.MediaPlaceholderMismatch;
    }
    return a.realloc(buf, n);
}

fn encodeText(ctx: ?*anyopaque, text: abi.Str, out: [*]u32, cap: u32) callconv(.c) u32 {
    const tok: *const Tokenizer = @ptrCast(@alignCast(ctx orelse return 0));
    const ids = tok.encode(std.heap.c_allocator, text.slice()) catch return 0;
    defer std.heap.c_allocator.free(ids);
    if (ids.len > cap) return 0;
    @memcpy(out[0..ids.len], ids);
    return @intCast(ids.len);
}

// ── tests ──

const testing = std.testing;

const Fake = struct {
    fn rows(p: *const abi.Parsed, item: *const abi.MediaItem, tok: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
        _ = tok;
        if (cap < item.rows + 2) return 0;
        out[0] = 7;
        @memset(out[1 .. 1 + item.rows], p.placeholder_ids.video);
        out[1 + item.rows] = 8;
        return item.rows + 2;
    }
    /// Claims more ids than the room it was given.
    fn overrun(p: *const abi.Parsed, item: *const abi.MediaItem, tok: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
        _ = .{ p, item, tok, out };
        return cap + 1;
    }
    fn short(p: *const abi.Parsed, item: *const abi.MediaItem, tok: *const abi.TextTokens, out: [*]u32, cap: u32) callconv(.c) u32 {
        _ = .{ p, tok, cap };
        out[0] = 7;
        return item.rows - 1;
    }
};

test "plugin media: a run is the plugin's, else rows of the kind's placeholder; one that does not hold the rows is refused" {
    const a = testing.allocator;
    var p: abi.Parsed = .{};
    p.placeholder_ids.image = 5;
    p.placeholder_ids.video = 6;
    const item: abi.MediaItem = .{ .kind = .video, .rows = 3 };
    var hooks: abi.MediaHooks = .{ .tower = undefined, .prepare = undefined };

    const image: abi.MediaItem = .{ .kind = .image, .rows = 2 };
    const plain = try runOf(a, &hooks, &p, &image, 2, null);
    defer a.free(plain);
    try testing.expectEqualSlices(u32, &.{ 5, 5 }, plain);

    hooks.placeholder_rows = &Fake.rows;
    const wrapped = try runOf(a, &hooks, &p, &item, 3, null);
    defer a.free(wrapped);
    try testing.expectEqualSlices(u32, &.{ 7, 6, 6, 6, 8 }, wrapped);

    try testing.expectError(error.MediaPlaceholderMismatch, runOf(a, &hooks, &p, &item, 4, null));
    hooks.placeholder_rows = &Fake.short;
    try testing.expectError(error.MediaPlaceholderMismatch, runOf(a, &hooks, &p, &item, 3, null));
    hooks.placeholder_rows = &Fake.overrun;
    try testing.expectError(error.MediaPlaceholderMismatch, runOf(a, &hooks, &p, &item, 3, null));
    const audio: abi.MediaItem = .{ .kind = .audio, .rows = 1 };
    try testing.expectError(error.MediaPlaceholderMismatch, runOf(a, &hooks, &p, &audio, 1, null));
}
