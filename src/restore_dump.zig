const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");

const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;

pub const RestoreDumpMeta = struct {
    kind: []const u8,
    pos: usize,
    cp: usize = 0,
    bank_from: usize = 0,
    source: []const u8 = "own",
    entry_idx: usize = 0,
    entry_count: usize = 0,
    mtp_base: ?usize = null,
};

pub var dump_restore_override: ?[]const u8 = null;
var dump_seq: std.atomic.Value(u64) = .init(0);

pub fn dumpRestoreDir() ?[]const u8 {
    if (dump_restore_override) |d| {
        if (d.len == 0 or d[0] == '0') return null;
        return d;
    }
    return dumpRestoreDirFromEnv(std.c.getenv("MLX_SERVE_DUMP_RESTORE"));
}

pub fn dumpRestoreDirFromEnv(raw: ?[*:0]const u8) ?[]const u8 {
    const v = raw orelse return null;
    if (v[0] == 0 or v[0] == '0') return null;
    return std.mem.span(v);
}

pub fn dumpRestoreIfEnabled(
    cache: *KVCache,
    ssm: ?[]SSMCacheEntry,
    s: mlx.mlx_stream,
    meta: RestoreDumpMeta,
) ?u64 {
    const dir = dumpRestoreDir() orelse return null;
    const n = dump_seq.fetchAdd(1, .monotonic) + 1;
    // A failed save raises inside mlx and our handler latches it; a diagnostic
    // must never turn into the request's MlxFailure, so drop what it set.
    const had_error = mlx.errorPending();
    dumpRestore(dir, n, cache, ssm, s, meta) catch |err| {
        log.warn("  [hot-cache] restore dump failed: {s}\n", .{@errorName(err)});
    };
    if (!had_error and mlx.errorPending()) {
        var buf: [512]u8 = undefined;
        _ = mlx.takeError(&buf);
    }
    return n;
}

fn dumpRestore(
    dir: []const u8,
    n: u64,
    cache: *KVCache,
    ssm: ?[]SSMCacheEntry,
    s: mlx.mlx_stream,
    meta: RestoreDumpMeta,
) !void {
    var dir_z_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir_z = zfmt(&dir_z_buf, "{s}", .{dir}) catch return error.NameTooLong;
    _ = std.c.mkdir(dir_z.ptr, @as(std.c.mode_t, 0o755));
    const gpa = std.heap.page_allocator;
    const stem = try std.fmt.allocPrint(gpa, "{s}/{s}-{d}-pos{d}", .{ dir, meta.kind, n, meta.pos });
    defer gpa.free(stem);
    const st_path = try std.fmt.allocPrintSentinel(gpa, "{s}.safetensors", .{stem}, 0);
    defer gpa.free(st_path);
    const json_path = try std.fmt.allocPrint(gpa, "{s}.json", .{stem});
    defer gpa.free(json_path);

    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    const mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    var key_buf: [96]u8 = undefined;

    try putInt(map, try zfmt(&key_buf, "pos", .{}), @intCast(meta.pos));
    if (ssm) |entries| {
        for (entries, 0..) |*e, i| {
            try putArr(map, try zfmt(&key_buf, "layers.{d}.conv_state", .{i}), e.conv_state, s);
            try putArr(map, try zfmt(&key_buf, "layers.{d}.ssm_state", .{i}), e.ssm_state, s);
            try putArr(map, try zfmt(&key_buf, "layers.{d}.aux_state", .{i}), e.aux_state, s);
            try putPooled(map, try zfmt(&key_buf, "layers.{d}.qsa_pooled", .{i}), e, s);
            try putPlePrev(map, try zfmt(&key_buf, "layers.{d}.ple_prev", .{i}), e);
            const hist: c_int = if (e.qsa_hist_rows > 0) e.qsa_hist_rows else 0;
            const held: c_int = if (e.aux_state.ctx != null and mlx.getShape(e.aux_state).len >= 2)
                mlx.getShape(e.aux_state)[1]
            else
                e.qsa_key_rows;
            const ring_start: c_int = if (held > 0 and hist >= held) hist - held else 0;
            try putInt(map, try zfmt(&key_buf, "layers.{d}.qsa_rows", .{i}), hist);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.qsa_key_rows", .{i}), e.qsa_key_rows);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.qsa_hist_rows", .{i}), hist);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.qsa_ratio", .{i}), e.qsa_ratio);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.ring_start", .{i}), ring_start);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.initialized", .{i}), if (e.initialized) 1 else 0);
            try putInt(map, try zfmt(&key_buf, "layers.{d}.ple_prev_valid", .{i}), if (e.ple_prev_valid) 1 else 0);
        }
    }
    for (cache.entries, 0..) |*entry, i| {
        if (!entry.initialized) continue;
        var view = cache.denseView(@intCast(i), s) catch continue;
        defer view.deinit();
        try putKvTail(map, try zfmt(&key_buf, "kv.{d}.k_tail", .{i}), view.k, s);
        try putKvTail(map, try zfmt(&key_buf, "kv.{d}.v_tail", .{i}), view.v, s);
    }

    try mlx.check(mlx.mlx_save_safetensors(st_path, map, mmap));

    const mtp_txt: []const u8 = if (meta.mtp_base) |b|
        try std.fmt.allocPrint(gpa, "{d}", .{b})
    else
        try gpa.dupe(u8, "null");
    defer gpa.free(mtp_txt);
    const json = try std.fmt.allocPrint(
        gpa,
        "{{\"kind\":\"{s}\",\"pos\":{d},\"cp\":{d},\"bank_from\":{d},\"source\":\"{s}\",\"entry\":{d},\"entry_count\":{d},\"mtp_base\":{s}}}\n",
        .{ meta.kind, meta.pos, meta.cp, meta.bank_from, meta.source, meta.entry_idx, meta.entry_count, mtp_txt },
    );
    defer gpa.free(json);
    var json_z_buf: [std.fs.max_path_bytes]u8 = undefined;
    const json_z = zfmt(&json_z_buf, "{s}", .{json_path}) catch return error.NameTooLong;
    const jf = std.c.fopen(json_z.ptr, "w") orelse return error.FileNotFound;
    defer _ = std.c.fclose(jf);
    if (std.c.fwrite(json.ptr, 1, json.len, jf) != json.len) return error.WriteFailed;
}

fn zfmt(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]u8 {
    const n = try std.fmt.bufPrint(buf[0 .. buf.len - 1], fmt, args);
    buf[n.len] = 0;
    return buf[0..n.len :0];
}

fn putInt(map: mlx.mlx_map_string_to_array, key: [:0]const u8, v: c_int) !void {
    const arr = mlx.mlx_array_new_int(v);
    defer _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_array_eval(arr);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, arr));
}

fn putArr(map: mlx.mlx_map_string_to_array, key: [:0]const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) !void {
    _ = s;
    if (arr.ctx == null or mlx.mlx_array_size(arr) == 0) return;
    _ = mlx.mlx_array_eval(arr);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, arr));
}

fn putPlePrev(map: mlx.mlx_map_string_to_array, key: [:0]const u8, e: *const SSMCacheEntry) !void {
    const shape = [_]c_int{@intCast(e.ple_prev.len)};
    const arr = mlx.mlx_array_new_data(@ptrCast(&e.ple_prev), &shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(arr);
    _ = mlx.mlx_array_eval(arr);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, arr));
}

fn putPooled(map: mlx.mlx_map_string_to_array, key: [:0]const u8, e: *const SSMCacheEntry, s: mlx.mlx_stream) !void {
    if (e.qsa_pooled.ctx == null) return;
    const ratio = @max(e.qsa_ratio, 1);
    const hist: c_int = if (e.qsa_hist_rows > 0) e.qsa_hist_rows else 0;
    const need = @divTrunc(hist, ratio);
    const ps = mlx.getShape(e.qsa_pooled);
    if (ps.len < 2) {
        try putArr(map, key, e.qsa_pooled, s);
        return;
    }
    const keep = if (need > 0 and need < ps[1]) need else ps[1];
    if (keep <= 0) return;
    if (keep == ps[1]) {
        try putArr(map, key, e.qsa_pooled, s);
        return;
    }
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ ps[0], keep, ps[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&sliced, e.qsa_pooled, &start, 3, &stop, 3, &strides, 3, s));
    _ = mlx.mlx_array_eval(sliced);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, sliced));
}

fn putKvTail(map: mlx.mlx_map_string_to_array, key: [:0]const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) !void {
    if (arr.ctx == null or mlx.mlx_array_size(arr) == 0) return;
    const sh = mlx.getShape(arr);
    if (sh.len != 4) {
        try putArr(map, key, arr, s);
        return;
    }
    const t = sh[2];
    if (t <= 0) return;
    const n: c_int = @min(64, t);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    const start = [_]c_int{ 0, 0, t - n, 0 };
    const stop = [_]c_int{ sh[0], sh[1], t, sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&tail, arr, &start, 4, &stop, 4, &strides, 4, s));
    _ = mlx.mlx_array_eval(tail);
    try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, tail));
}

test "dumpRestoreDirFromEnv: absent or 0 is off" {
    try std.testing.expect(dumpRestoreDirFromEnv(null) == null);
    try std.testing.expect(dumpRestoreDirFromEnv("0") == null);
    try std.testing.expect(dumpRestoreDirFromEnv("") == null);
    const on = dumpRestoreDirFromEnv("/tmp/restore-dump");
    try std.testing.expect(on != null);
    try std.testing.expectEqualStrings("/tmp/restore-dump", on.?);
}
