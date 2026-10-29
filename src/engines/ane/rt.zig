//! The Neural Engine runtime under the ANE LM engine: compiled programs (`lib/ane` program API),
//! IOSurface buffers laid out as the compiled model wants them, and requests that bind a program's
//! inputs and outputs to buffers by name.
//!
//! Layout: a MIL tensor of rank r maps onto the ANE's (batch, channel, height, width) from the right
//! (the last axis is the width, packed; then rows, then planes, then batches), each with the byte
//! stride the compiled model reports. Rows are padded to 64 bytes, so a [1, 5120, 1, 8] FP16 value
//! takes 64 bytes per channel.

const std = @import("std");
const env = @import("../../env.zig");
const builtin = @import("builtin");

pub const Io = extern struct {
    name: [128]u8,
    dtype: u32,
    batches: u32,
    channels: u32,
    depth: u32,
    height: u32,
    width: u32,
    batch_stride: u64,
    depth_stride: u64,
    plane_stride: u64,
    row_stride: u64,

    pub fn nameSlice(self: *const Io) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }

    pub fn bytes(self: Io) usize {
        return @intCast(self.batch_stride * @max(self.batches, 1));
    }
};

const Program_ = opaque {};
const Request_ = opaque {};
const Surface_ = opaque {};

extern fn msv_ane_bridge_available() c_int;
extern fn msv_ane_bridge_surface(bytes: usize) ?*Surface_;
extern fn msv_ane_program_create(name: [*:0]const u8, mil: [*:0]const u8, weights: ?*anyopaque, weight_bytes: usize, instances: u32, ane_hints: ?[*]const u32, err: [*]u8, err_size: usize) ?*Program_;
extern fn msv_ane_program_free(p: ?*Program_) void;
extern fn msv_ane_program_procedure(p: *Program_, function: [*:0]const u8) c_int;
extern fn msv_ane_program_io_count(p: *Program_, proc: u32, output: c_int) u32;
extern fn msv_ane_program_io(p: *Program_, proc: u32, output: c_int, index: u32, out: *Io) c_int;
extern fn msv_ane_request_create(p: *Program_, instance: u32, proc: u32, inputs: [*]const ?*Surface_, n_in: u32, outputs: [*]const ?*Surface_, n_out: u32, err: [*]u8, err_size: usize) ?*Request_;
extern fn msv_ane_request_eval(r: *Request_, err: [*]u8, err_size: usize) c_int;
extern fn msv_ane_request_free(r: ?*Request_) void;
extern fn msv_ane_program_compile_seconds(p: ?*const Program_) f64;
extern fn msv_ane_program_cache_hit(p: ?*const Program_) bool;
extern fn IOSurfaceGetBaseAddress(s: *Surface_) ?*anyopaque;
extern fn IOSurfaceGetAllocSize(s: *Surface_) usize;
extern fn CFRelease(cf: *anyopaque) void;

pub const Error = error{ AneUnavailable, CompileFailed, NoSuchFunction, BindFailed, RunFailed, LayoutMismatch, OutOfMemory };

pub fn available() bool {
    if (comptime !builtin.os.tag.isDarwin()) return false;
    return msv_ane_bridge_available() != 0;
}

pub const ErrBuf = struct {
    buf: [1024]u8 = @splat(0),
    pub fn text(self: *const ErrBuf) []const u8 {
        return std.mem.sliceTo(&self.buf, 0);
    }
};

pub const DType = enum(u32) { fp16 = 0, fp32 = 1, int8 = 2, uint8 = 3, _ };

/// A view of one buffer in the compiled model's layout. `dims` are the MIL shape; `strides` the byte
/// stride of each MIL axis (the last is the element size).
pub const Tensor = struct {
    base: [*]u8,
    dtype: DType,
    rank: u8,
    dims: [4]u32,
    strides: [4]usize,
    bytes: usize,

    pub fn elemSize(self: Tensor) usize {
        return switch (self.dtype) {
            .fp16 => 2,
            .fp32 => 4,
            else => 1,
        };
    }

    /// The tensor that `io` describes, read as MIL shape `dims` (rank <= 4).
    pub fn of(base: [*]u8, io: Io, dims: []const u32) Error!Tensor {
        var t: Tensor = .{ .base = base, .dtype = @enumFromInt(io.dtype), .rank = @intCast(dims.len), .dims = @splat(1), .strides = @splat(0), .bytes = io.bytes() };
        @memcpy(t.dims[0..dims.len], dims);
        const lanes = [4]u64{ io.batches, io.channels, io.height, io.width };
        const strides = [4]u64{ io.batch_stride, io.plane_stride, io.row_stride, t.elemSize() };
        var count: u64 = 1;
        for (dims) |d| count *= d;
        if (count != @as(u64, io.batches) * io.channels * io.depth * io.height * io.width) return error.LayoutMismatch;
        // Right-aligned: MIL axis i is ANE lane 4 - rank + i.
        for (0..dims.len) |i| {
            const lane = 4 - dims.len + i;
            if (lanes[lane] != dims[i]) return error.LayoutMismatch;
            t.strides[i] = @intCast(strides[lane]);
        }
        return t;
    }

    pub fn offset(self: Tensor, idx: []const u32) usize {
        var o: usize = 0;
        for (idx, 0..) |x, i| o += @as(usize, x) * self.strides[i];
        return o;
    }

    pub fn f16At(self: Tensor, idx: []const u32) *align(1) f16 {
        return @ptrCast(self.base + self.offset(idx));
    }

    pub fn zero(self: Tensor) void {
        @memset(self.base[0..self.bytes], 0);
    }

    /// The contiguous run of the last axis at `idx` (whose last entry is ignored).
    pub fn row(self: Tensor, comptime T: type, idx: []const u32) []align(1) T {
        var at: [4]u32 = @splat(0);
        @memcpy(at[0 .. self.rank - 1], idx[0 .. self.rank - 1]);
        const p: [*]align(1) T = @ptrCast(self.base + self.offset(at[0..self.rank]));
        return p[0..self.dims[self.rank - 1]];
    }
};

/// One IOSurface: the buffer of one named input or output, shared by every program that names it.
pub const Surface = struct {
    ref: *Surface_,
    base: [*]u8,
    bytes: usize,

    pub fn create(bytes: usize) Error!Surface {
        if (comptime !builtin.os.tag.isDarwin()) return error.AneUnavailable;
        const s = msv_ane_bridge_surface(bytes) orelse return error.OutOfMemory;
        const base: [*]u8 = @ptrCast(IOSurfaceGetBaseAddress(s) orelse return error.OutOfMemory);
        const n = IOSurfaceGetAllocSize(s);
        @memset(base[0..n], 0);
        return .{ .ref = s, .base = base, .bytes = n };
    }

    pub fn release(self: Surface) void {
        if (comptime builtin.os.tag.isDarwin()) CFRelease(self.ref);
    }
};

pub const Program = struct {
    handle: *Program_,

    /// Compile (or restore from the compile cache) and load one instance per entry of `hints`, each
    /// pinned to that ANE instance (0 = the system picks). Takes the weight blob (allocated with
    /// `std.heap.c_allocator`, freed by the bridge).
    pub fn create(name: [:0]const u8, mil: [:0]const u8, weights: []u8, hints: []const u32, err: *ErrBuf) Error!Program {
        if (!available()) return error.AneUnavailable;
        const h = msv_ane_program_create(name.ptr, mil.ptr, if (weights.len > 0) weights.ptr else null, weights.len, @intCast(hints.len), hints.ptr, &err.buf, err.buf.len) orelse return error.CompileFailed;
        return .{ .handle = h };
    }

    pub fn deinit(self: Program) void {
        msv_ane_program_free(self.handle);
    }

    pub fn function(self: Program, name: [:0]const u8) Error!u32 {
        const p = msv_ane_program_procedure(self.handle, name.ptr);
        return if (p < 0) error.NoSuchFunction else @intCast(p);
    }

    pub fn ioCount(self: Program, proc: u32, output: bool) u32 {
        return msv_ane_program_io_count(self.handle, proc, @intFromBool(output));
    }

    pub fn io(self: Program, proc: u32, output: bool, i: u32) Error!Io {
        var out: Io = undefined;
        if (msv_ane_program_io(self.handle, proc, @intFromBool(output), i, &out) == 0) return error.BindFailed;
        return out;
    }

    pub fn compileSeconds(self: Program) f64 {
        return msv_ane_program_compile_seconds(self.handle);
    }

    pub fn cacheHit(self: Program) bool {
        return msv_ane_program_cache_hit(self.handle);
    }

    /// A request for `proc` on `instance`, its i-th input bound to `inputs[i]` (in `io` order).
    pub fn request(self: Program, instance: u32, proc: u32, inputs: []const ?*Surface_, outputs: []const ?*Surface_, err: *ErrBuf) Error!Request {
        const r = msv_ane_request_create(self.handle, instance, proc, inputs.ptr, @intCast(inputs.len), outputs.ptr, @intCast(outputs.len), &err.buf, err.buf.len) orelse return error.BindFailed;
        return .{ .handle = r };
    }
};

pub const Request = struct {
    handle: *Request_,

    pub fn run(self: Request, err: *ErrBuf) Error!void {
        if (msv_ane_request_eval(self.handle, &err.buf, err.buf.len) == 0) return error.RunFailed;
    }

    pub fn deinit(self: Request) void {
        msv_ane_request_free(self.handle);
    }
};

pub const SurfaceRef = ?*Surface_;

/// Tag the compile-cache entries the next programs restore or store with (`group`, `variant`): the
/// cache never evicts that set while it builds, and drops the group's other variants first. Empty
/// strings untag.
pub fn cacheLineage(group: [:0]const u8, variant: [:0]const u8) void {
    if (comptime builtin.os.tag.isDarwin()) msv_ane_cache_lineage(group.ptr, variant.ptr);
}

/// After a build under the current lineage: its group's entries it did not restore or store go.
pub fn cacheSweep() void {
    if (comptime builtin.os.tag.isDarwin()) msv_ane_cache_sweep();
}

// ── tests ──

const testing = std.testing;

test "tensor layout: MIL axes map onto (batch, plane, row, width) from the right" {
    var io: Io = std.mem.zeroes(Io);
    io.dtype = 0;
    io.batches = 1;
    io.channels = 4;
    io.depth = 1;
    io.height = 8;
    io.width = 16;
    io.row_stride = 64;
    io.plane_stride = 512;
    io.batch_stride = 2048;
    var buf: [2048]u8 align(16) = undefined;
    const t = try Tensor.of(&buf, io, &.{ 4, 8, 16 });
    try testing.expectEqual(@as(usize, 512), t.strides[0]);
    try testing.expectEqual(@as(usize, 64), t.strides[1]);
    try testing.expectEqual(@as(usize, 2), t.strides[2]);
    try testing.expectEqual(@as(usize, 2 * 512 + 3 * 64 + 5 * 2), t.offset(&.{ 2, 3, 5 }));
    // A [1, C, 1, W] activation: the channel is the plane.
    io.channels = 64;
    io.height = 1;
    io.width = 8;
    io.plane_stride = 64;
    io.batch_stride = 64 * 64;
    var big: [4096]u8 align(16) = undefined;
    const x = try Tensor.of(&big, io, &.{ 1, 64, 1, 8 });
    try testing.expectEqual(@as(usize, 64 * 10 + 2 * 3), x.offset(&.{ 0, 10, 0, 3 }));
    try testing.expectError(error.LayoutMismatch, Tensor.of(&big, io, &.{ 64, 8 * 2 }));
}

extern fn msv_ane_cache_lineage(group: [*:0]const u8, variant: [*:0]const u8) void;
extern fn msv_ane_cache_prune() void;
extern fn msv_ane_cache_sweep() void;

/// One cache entry of `gb` GiB (sparse) at `mtime`, tagged `group`/`variant` unless empty.
fn testEntry(root: []const u8, name: []const u8, gb: i64, mtime: i64, lineage: []const u8) !void {
    var b: [512]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&b, "{s}/entries/{s}", .{ root, name }, 0);
    if (std.c.mkdir(dir, 0o755) != 0) return error.Unexpected;
    var fb: [512]u8 = undefined;
    for ([_][]const u8{ "data", "compiled.ok", "lineage" }) |f| {
        if (std.mem.eql(u8, f, "lineage") and lineage.len == 0) continue;
        const path = try std.fmt.bufPrintSentinel(&fb, "{s}/{s}", .{ dir, f }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.Unexpected;
        defer _ = std.c.close(fd);
        if (std.mem.eql(u8, f, "data")) _ = std.c.ftruncate(fd, gb << 30);
        if (std.mem.eql(u8, f, "lineage")) _ = std.c.write(fd, lineage.ptr, lineage.len);
    }
    var tv = [2]std.c.timeval{ .{ .sec = mtime, .usec = 0 }, .{ .sec = mtime, .usec = 0 } };
    _ = std.c.utimes(dir, &tv);
}

test "cache sweep: after a build, its group's entries it did not touch go; other groups stay" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var rb: [128]u8 = undefined;
    const root = try std.fmt.bufPrintSentinel(&rb, "/tmp/msv-ane-sweep-{d}", .{std.c.getpid()}, 0);
    var eb: [160]u8 = undefined;
    _ = std.c.mkdir(root, 0o755);
    _ = std.c.mkdir(try std.fmt.bufPrintSentinel(&eb, "{s}/entries", .{root}, 0), 0o755);
    defer std.Io.Dir.cwd().deleteTree(testing.io, root) catch {};
    env.set(.MLX_SERVE_ANE_CACHE_DIR, root);
    defer {
        env.unset(.MLX_SERVE_ANE_CACHE_DIR);
        msv_ane_cache_lineage("", "");
    }
    msv_ane_cache_lineage("lm:a", "v1");
    // Touched after the lineage was named, as a restore or a store touches it.
    try testEntry(root, "used", 0, 4_000_000_000, "lm:a\nv1\n");
    try testEntry(root, "dead", 0, 1_000, "lm:a\nv1\n");
    try testEntry(root, "other", 0, 1_000, "lm:b\nv1\n");
    try testEntry(root, "untagged", 0, 1_000, "");
    msv_ane_cache_sweep();
    for ([_][]const u8{ "used", "dead", "other", "untagged" }, [_]bool{ true, false, true, true }) |name, kept| {
        var pb: [256]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&pb, "{s}/entries/{s}/compiled.ok", .{ root, name }, 0);
        try testing.expectEqual(kept, std.c.access(p, 0) == 0);
    }
}

test "cache prune: never the set being built; its group's other variants first, then the oldest" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var rb: [128]u8 = undefined;
    const root = try std.fmt.bufPrintSentinel(&rb, "/tmp/msv-ane-prune-{d}", .{std.c.getpid()}, 0);
    var eb: [160]u8 = undefined;
    _ = std.c.mkdir(root, 0o755);
    _ = std.c.mkdir(try std.fmt.bufPrintSentinel(&eb, "{s}/entries", .{root}, 0), 0o755);
    defer std.Io.Dir.cwd().deleteTree(testing.io, root) catch {};
    try testEntry(root, "current", 1, 1_000, "lm:a\nv1\n");
    try testEntry(root, "stale", 1, 4_000, "lm:a\nv0\n");
    try testEntry(root, "older", 1, 2_000, "");
    try testEntry(root, "newer", 1, 3_000, "");
    env.set(.MLX_SERVE_ANE_CACHE_DIR, root);
    env.set(.MLX_SERVE_ANE_CACHE_CAP_GB, "3");
    defer {
        env.unset(.MLX_SERVE_ANE_CACHE_DIR);
        env.unset(.MLX_SERVE_ANE_CACHE_CAP_GB);
        msv_ane_cache_lineage("", "");
    }
    msv_ane_cache_lineage("lm:a", "v1");
    msv_ane_cache_prune();
    for ([_][]const u8{ "current", "stale", "older", "newer" }, [_]bool{ true, false, false, true }) |name, kept| {
        var pb: [256]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&pb, "{s}/entries/{s}/compiled.ok", .{ root, name }, 0);
        try testing.expectEqual(kept, std.c.access(p, 0) == 0);
    }
}
