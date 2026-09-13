//! Opt-in exact PLE gather. Only selected affine4 rows enter the MLX graph.
const std = @import("std");
const mlx = @import("mlx.zig");
const qwen = @import("qwen4_exp.zig");
const log = @import("log.zig");
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var announced = false;

pub fn eligible(table: *const qwen.NgramTable, n: usize, deferred: bool) bool {
    const on = @import("prefill_experiment.zig").ple() orelse blk: {
        const raw = std.c.getenv("MLX_SERVE_PLE_PACKED") orelse break :blk false;
        break :blk raw[0] == '1';
    };
    return on and !mlx.noGpuBackend() and !deferred and n >= 64 and n <= 8192 and supported(table);
}
fn supported(t: *const qwen.NgramTable) bool {
    return t.rows <= std.math.maxInt(u32) and t.bits == 4 and t.group_size == 32 and t.dim == 160 and t.wcols == 20 and t.scols == 5;
}
const Packed = struct {
    data: []u8,
    slots: []u32,
    used: usize,
    fn deinit(self: Packed, a: std.mem.Allocator) void {
        a.free(self.data);
        a.free(self.slots);
    }
};
fn pack(a: std.mem.Allocator, table: *const qwen.NgramTable, rows: []const i64) !Packed {
    return packCached(a, table, rows, null);
}
const RowCache = struct {
    const capacity = 262144;
    data: []u8,
    index: std.AutoHashMapUnmanaged(u32, u32) = .empty,
    hits: usize = 0,
    misses: usize = 0,
    fn init(a: std.mem.Allocator) !RowCache {
        var self = RowCache{ .data = try a.alloc(u8, capacity * 100) };
        errdefer a.free(self.data);
        try self.index.ensureTotalCapacity(a, capacity);
        return self;
    }
    fn deinit(self: *RowCache, a: std.mem.Allocator) void {
        self.index.deinit(a);
        a.free(self.data);
    }
};
fn packCached(a: std.mem.Allocator, table: *const qwen.NgramTable, rows: []const i64, cache: ?*RowCache) !Packed {
    if (!supported(table) or rows.len == 0 or rows.len > 8192 * qwen.MAX_HEADS) return error.UnsupportedPlePacked;
    const data = try a.alloc(u8, rows.len * 100);
    errdefer a.free(data);
    const slots = try a.alloc(u32, rows.len);
    errdefer a.free(slots);
    var unique: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer unique.deinit(a);
    try unique.ensureTotalCapacity(a, @intCast(rows.len));
    var used: usize = 0;
    for (rows, slots) |id, *slot| {
        if (id < 0 or @as(u64, @intCast(id)) >= table.rows) return error.InvalidPleRow;
        const result = unique.getOrPutAssumeCapacity(@intCast(id));
        if (!result.found_existing) {
            result.value_ptr.* = @intCast(used);
            const dst = data[used * 100 ..][0..100];
            const row: usize = @intCast(id);
            const hit = if (cache) |c| c.index.get(@intCast(id)) else null;
            if (hit) |at| {
                @memcpy(dst, cache.?.data[@as(usize, at) * 100 ..][0..100]);
                cache.?.hits += 1;
            } else {
                @memcpy(dst[0..80], table.map[table.w_off + row * 80 ..][0..80]);
                @memcpy(dst[80..90], table.map[table.s_off + row * 10 ..][0..10]);
                @memcpy(dst[90..100], table.map[table.b_off + row * 10 ..][0..10]);
                if (cache) |c| {
                    c.misses += 1;
                    if (c.index.count() < RowCache.capacity) {
                        const at = c.index.count();
                        @memcpy(c.data[@as(usize, at) * 100 ..][0..100], dst);
                        c.index.putAssumeCapacity(@intCast(id), at);
                    }
                }
            }
            used += 1;
        }
        slot.* = result.value_ptr.*;
    }
    return .{ .data = data, .slots = slots, .used = used };
}
fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const names = [_][*:0]const u8{ "packed", "slots" };
    const ins = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(&[_][*:0]const u8{"out"}, 1);
    defer _ = mlx.mlx_vector_string_free(outs);
    const k = mlx.mlx_fast_metal_kernel_new("msv_ple_packed", ins, outs, @embedFile("kernels/ple_packed.metal"), "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}
pub fn gather(a: std.mem.Allocator, s: mlx.mlx_stream, table: *const qwen.NgramTable, rows: []const i64) !mlx.mlx_array {
    const clock_io = std.Io.Threaded.global_single_threaded.io();
    const t = std.Io.Timestamp.now(clock_io, .boot);
    const host = try pack(a, table, rows);
    defer host.deinit(a);
    if (std.c.getenv("MLX_SERVE_PLE_TIMING") != null) log.info("[ple-packed] host pack: rows={d} unique={d} ms={d:.2}\n", .{ rows.len, host.used, @as(f64, @floatFromInt(t.untilNow(clock_io, .boot).nanoseconds)) / 1e6 });
    return upload(s, host);
}
fn upload(s: mlx.mlx_stream, host: Packed) !mlx.mlx_array {
    const row_count = host.slots.len;
    // new_data copies into owned MLX arrays. The lazy graph cannot observe a
    // later chunk's scratch or a mutated cache entry, even after this returns.
    const p = mlx.mlx_array_new_data(host.data.ptr, &[_]c_int{ @intCast(host.used), 100 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(p);
    const ids = mlx.mlx_array_new_data(host.slots.ptr, &[_]c_int{@intCast(row_count)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(row_count), 160 }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(row_count * 160), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(&.{ p, ids }, 2);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(), ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    if (!announced and row_count >= 8192) {
        announced = true;
        log.info("[ple-packed] engaged: rows={d} unique={d} packed_bytes={d} immutable=true output=mlx-bf16\n", .{ row_count, host.used, host.used * 100 });
    }
    return out;
}

/// Request-owned CPU producer. The GPU thread receives immutable selected-row
/// snapshots; no MLX calls, token-history mutations or model state on the worker.
/// At most two ready chunks, one in preparation, and a 26.2 MB raw-row cache.
pub const Ahead = struct {
    table: *const qwen.NgramTable,
    rows: []i64,
    ends: []usize,
    thread: ?std.Thread = null,
    mu: std.Io.Mutex = .init,
    cv: std.Io.Condition = .init,
    stop: bool = false,
    done: bool = false,
    read: usize = 0,
    written: usize = 0,
    ring: [2]?Packed = .{ null, null },
    pack_ns: u64 = 0,
    wait_ns: u64 = 0,
    hits: usize = 0,
    misses: usize = 0,
    const a = std.heap.page_allocator;
    fn io() std.Io {
        return std.Io.Threaded.global_single_threaded.io();
    }

    pub fn enabled() bool {
        if (@import("prefill_experiment.zig").ple()) |v| return v;
        const v = std.c.getenv("MLX_SERVE_PLE_AHEAD") orelse return false;
        return v[0] == '1';
    }
    /// Clone all input metadata before spawning. `ends` are cumulative row
    /// offsets at the actual, fixed prefill chunk boundaries.
    pub fn create(table: *const qwen.NgramTable, rows: []const i64, ends: []const usize) !*Ahead {
        if (!supported(table) or ends.len == 0 or rows.len > 131072 * qwen.MAX_HEADS) return error.UnsupportedPleAhead;
        var from: usize = 0;
        for (ends) |end| {
            if (end <= from or end > rows.len or end - from > 8192 * qwen.MAX_HEADS) return error.UnsupportedPleAhead;
            from = end;
        }
        if (from != rows.len) return error.UnsupportedPleAhead;
        const self = try a.create(Ahead);
        errdefer a.destroy(self);
        const r = try a.dupe(i64, rows);
        errdefer a.free(r);
        const e = try a.dupe(usize, ends);
        errdefer a.free(e);
        self.* = .{ .table = table, .rows = r, .ends = e };
        self.thread = try std.Thread.spawn(.{}, worker, .{self});
        return self;
    }
    fn worker(self: *Ahead) void {
        var cache = RowCache.init(a) catch {
            self.finish();
            return;
        };
        defer {
            self.hits = cache.hits;
            self.misses = cache.misses;
            cache.deinit(a);
            self.finish();
        }
        var from: usize = 0;
        for (self.ends) |end| {
            self.mu.lockUncancelable(io());
            while (self.written - self.read == self.ring.len and !self.stop) self.cv.wait(io(), &self.mu) catch {};
            const quit = self.stop;
            self.mu.unlock(io());
            if (quit) return;
            const t = std.Io.Timestamp.now(io(), .boot);
            const p = packCached(a, self.table, self.rows[from..end], &cache) catch return;
            self.pack_ns += @intCast(t.untilNow(io(), .boot).nanoseconds);
            self.mu.lockUncancelable(io());
            if (self.stop) {
                self.mu.unlock(io());
                p.deinit(a);
                return;
            }
            self.ring[self.written % self.ring.len] = p;
            self.written += 1;
            self.cv.broadcast(io());
            self.mu.unlock(io());
            from = end;
        }
    }
    fn finish(self: *Ahead) void {
        self.mu.lockUncancelable(io());
        self.done = true;
        self.cv.broadcast(io());
        self.mu.unlock(io());
    }
    fn take(self: *Ahead, expected_rows: []const i64) ?Packed {
        const t = std.Io.Timestamp.now(io(), .boot);
        self.mu.lockUncancelable(io());
        defer self.mu.unlock(io());
        // Compare every row ID, including history-dependent first n-grams.
        // An unexpected forward safely declines instead of consuming stale data.
        const from = if (self.read == 0) 0 else self.ends[self.read - 1];
        if (self.stop or self.read >= self.ends.len or !std.mem.eql(i64, expected_rows, self.rows[from..self.ends[self.read]])) {
            self.stop = true;
            self.cv.broadcast(io());
            return null;
        }
        while (self.read == self.written and !self.done and !self.stop) self.cv.wait(io(), &self.mu) catch {};
        self.wait_ns += @intCast(t.untilNow(io(), .boot).nanoseconds);
        if (self.read == self.written) return null;
        const slot = self.read % self.ring.len;
        const p = self.ring[slot].?;
        self.ring[slot] = null;
        self.read += 1;
        self.cv.broadcast(io());
        return p;
    }
    pub fn gather(self: *Ahead, s: mlx.mlx_stream, rows: []const i64) !?mlx.mlx_array {
        const p = self.take(rows) orelse return null;
        defer p.deinit(a);
        return try upload(s, p);
    }
    pub fn destroy(self: *Ahead) void {
        self.mu.lockUncancelable(io());
        self.stop = true;
        self.cv.broadcast(io());
        self.mu.unlock(io());
        if (self.thread) |th| th.join();
        for (self.ring) |p| if (p) |v| v.deinit(a);
        log.info("[ple-ahead] chunks={d}/{d} pack={d:.2}ms wait={d:.2}ms row_hits={d} row_misses={d} cache_limit_rows={d}\n", .{ self.read, self.ends.len, @as(f64, @floatFromInt(self.pack_ns)) / 1e6, @as(f64, @floatFromInt(self.wait_ns)) / 1e6, self.hits, self.misses, RowCache.capacity });
        a.free(self.rows);
        a.free(self.ends);
        a.destroy(self);
    }
};

test "PLE packed exact native rows, duplicates, tail and deferred input ownership" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const map = try a.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 7 * 100);
    defer a.free(map);
    const t = qwen.NgramTable{ .map = map, .rows = 7, .dim = 160, .bits = 4, .group_size = 32, .w_off = 0, .s_off = 7 * 80, .b_off = 7 * 90, .wcols = 20, .scols = 5 };
    for (0..7 * 20) |i| std.mem.writeInt(u32, map[i * 4 ..][0..4], @truncate(i *% 0x9e3779b9 +% 0x12345678), .little);
    for (0..35) |i| {
        std.mem.writeInt(u16, map[t.s_off + i * 2 ..][0..2], @as(u16, 0x3b31) + @as(u16, @intCast(i * 13)), .little);
        std.mem.writeInt(u16, map[t.b_off + i * 2 ..][0..2], @as(u16, 0xbf07) + @as(u16, @intCast(i * 7)), .little);
    }
    var rows: [257]i64 = undefined;
    for (&rows, 0..) |*id, i| id.* = @intCast((i * 3) % 7);
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const x = try gather(a, s, &t, &rows);
    defer _ = mlx.mlx_array_free(x);
    // Create another pending graph before evaluating the first; its temporary
    // host buffers must not replace the first graph's inputs.
    const y = try gather(a, s, &t, &.{ 6, 0, 6 });
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    try mlx.check(mlx.mlx_array_eval(x));
    const got = mlx.mlx_array_data_bfloat16(x).?;
    var native: [160]f32 = undefined;
    for (rows, 0..) |id, r| {
        t.row(@intCast(id), &native);
        for (native, 0..) |v, c| {
            const bits: u32 = @bitCast(v);
            const expected: u16 = @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
            try std.testing.expectEqual(expected, got[r * 160 + c]);
        }
    }
    try std.testing.expectError(error.InvalidPleRow, pack(a, &t, &.{-1}));
    try std.testing.expectError(error.InvalidPleRow, pack(a, &t, &.{7}));

    const ahead = try Ahead.create(&t, &rows, &.{ 64, 192, 257 });
    defer ahead.destroy();
    var from: usize = 0;
    for ([_]usize{ 64, 192, 257 }) |end| {
        const result = (try ahead.gather(s, rows[from..end])) orelse return error.AheadDeclined;
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_array_eval(result));
        const values = mlx.mlx_array_data_bfloat16(result).?;
        try std.testing.expectEqualSlices(u16, got[from * 160 .. end * 160], values[0 .. (end - from) * 160]);
        from = end;
    }
    try std.testing.expect(ahead.take(&.{}) == null);

    // A mismatched history/forward stops the producer and safely declines.
    const mismatch = try Ahead.create(&t, &rows, &.{ 64, 192, 257 });
    try std.testing.expect(mismatch.take(&.{6}) == null);
    mismatch.destroy();
    // Cancellation without draining a possibly full ring must join cleanly.
    const cancelled = try Ahead.create(&t, &rows, &.{ 64, 192, 257 });
    cancelled.destroy();
}
