//! The per-prompt regions the wide prompt layers' arena levers write their large transients into, not fresh MLX
//! allocations: `Region` and the cap on their bytes together. Users:
//! scratch_x1.zig (a wide layer's streams after its attention) and scratch_attn.zig.
//!
//! A 16K-row encoder layer allocated ~21 GB of fresh Metal buffers (pt2: MLX's buffer cache is capped at 2 GiB, so each
//! layer's GB-sized transients are new `newBufferWithLength` calls, ~15 ms/GB of allocator, residency and fresh pages).
//! A `Region` is one MTLBuffer from the host's `alloc_uninit` (bytes unspecified until written), allocated at its first
//! use in a prompt and kept while its shape holds, freed at the decode phase change (its lever's `releaseAll`). Never a
//! view of a bank or window array, never shared by two tensors that live at once.
//!
//! MLX's custom kernels allocate every output, so a writer writes the region as an input it casts to `device T*` (a
//! whole region or an axis-0 slice of one: row-contiguous, so `ensure_row_contiguous` never copies it) and declares a
//! one-word `carrier` output (`Region.wrote` keeps the latest). MLX tracks hazards per MTLBuffer, through registered
//! outputs only: a reader is ordered after a writer only through the writer's carrier. A reader of our own binds the
//! carrier as one more input it never reads (`dep`): MLX's barrier (one compute pass) or fence (a later pass, or a later
//! command buffer) orders it. A reader that cannot (an MLX op, a kernel left as it is) takes the region through
//! `Region.publish`: the host's event wait over an event already at 1 with the region and the carrier as its tracked
//! inputs; the wait ends the compute pass, and the pass after it binds the carrier, so MLX fences that pass on the
//! writer's, and its outputs alias the region's buffer and register as that pass's, so every later reader is fenced on
//! it. (Untracked dependencies would not do: an event wait that is already satisfied orders nothing, and MLX fences a
//! pass only on buffers bound in it.) One compute-pass split a publish.
//!
//! Writers of one region between two reads are ordered among themselves (each binds the previous carrier as its
//! `after`, or a host drain separates them): `wrote` keeps the latest carrier only.
//!
//! The ordering a writer needs before it overwrites a region (its earlier readers done) is not MLX's: MLX never reuses a
//! buffer before its command buffer completes, so it tracks no write-after-read across command buffers. The caller hands
//! the region an output of its last reader (`Region.released`), and the next writer binds it as an input (`after`), so
//! MLX orders the writer behind it; a use whose next writer already depends on it, or is behind a host drain, may skip it.
//! Exact: the same kernels and arithmetic write the same words; only where they land changes.
//!
//! The regions' bytes together stay within `cap_bytes`; a region that would pass it is not made and its user takes the
//! fresh path.

const std = @import("std");
/// The regions ride the host's MLX shims (`sdk/weights/streamed/csrc/`), which only the macOS graphs compile: elsewhere
/// every caller takes the fresh path.
const shimmed = @import("builtin").os.tag == .macos;
const mlx = @import("../../../mlx/ffi.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

/// The host's C shims (`sdk/weights/streamed/csrc/mlx_alloc_shim.cpp`, `mlx_event_shim.cpp`): an evaluated array over one fresh buffer, bytes unspecified; the event wait.
extern fn dsv41_alloc_uninit(out: *A, shape: [*]const c_int, ndim: usize, dtype: mlx.mlx_dtype) c_int;
extern fn dsv41ev_create_metal(start: u64, object: *u64) i32;
extern fn dsv41ev_wait(xs: [*]const A, n: usize, event: i32, value: u64, deps: ?[*]const A, n_deps: usize, track_inputs: bool, s: mlx.mlx_stream, outs: [*]A) c_int;

/// An event already at 1: `publish`'s waits pass at once (only their ordering and aliasing matter).
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    ready_event: i32 = -1,
    held_bytes: u64 = 0,
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_scratch", .{});
pub inline fn ps() *State {
    return shared_state.get();
}
fn readyEvent() !i32 {
    if (comptime !shimmed) return error.ScratchEvent;
    if (ps().ready_event < 0) {
        var object: u64 = 0;
        const id = dsv41ev_create_metal(1, &object);
        if (id < 0) return error.ScratchEvent;
        ps().ready_event = id;
    }
    return ps().ready_event;
}

pub const max_ndim = 4;
/// The bank pool (bank_pool.zig): an array the stream's pool holds for a region, by shape and dtype
/// (scratch_waves.zig's wave sets').
pub const Shape = struct { shape: [max_ndim]c_int = @splat(0), ndim: usize = 0, dtype: mlx.mlx_dtype = .float32 };
/// The most bytes the regions hold together (a machine bound for the M5 Max's 128 GB beside the pack, never a prompt's).
pub const cap_bytes: u64 = 4 << 30;

/// One per-prompt buffer that writer kernels fill in place and readers take through their carrier or `publish`.
pub const Region = struct {
    arr: A = none,
    shape: [max_ndim]c_int = @splat(0),
    ndim: usize = 0,
    dtype: mlx.mlx_dtype = .float32,
    bytes: u64 = 0,
    /// The latest writer's carrier (a reference).
    carrier: A = none,
    /// An output of the last reader of the region's words (a reference): the next writer binds it (`after`).
    release: A = none,
    /// The bank pool: the pool's array for the region (a reference) and its shape: `ensure` takes it, not a fresh
    /// buffer, whenever its shape is the one asked.
    pooled: A = none,
    pooled_shape: Shape = .{},

    /// The region's buffer at `shape` of `dtype`: the one it holds while they match, else a new one (the old freed);
    /// null when the regions together would pass `cap_bytes` (the caller takes the fresh path).
    pub fn ensure(r: *Region, shape: []const c_int, dtype: mlx.mlx_dtype) !?A {
        if (comptime !shimmed) return null;
        if (shape.len == 0 or shape.len > max_ndim) return error.ScratchShape;
        if (r.arr.ctx != null and r.ndim == shape.len and std.mem.eql(c_int, r.shape[0..r.ndim], shape) and r.dtype == dtype) return r.arr;
        r.free();
        var n: u64 = dtypeBytes(dtype);
        for (shape) |d| n *= @intCast(d);
        if (ps().held_bytes + n > cap_bytes) return null;
        var out = mlx.mlx_array_new();
        // The bank pool: the pool's array when its shape is the one asked (a reference, nothing allocated).
        const pooled = r.pooled.ctx != null and r.pooled_shape.ndim == shape.len and std.mem.eql(c_int, r.pooled_shape.shape[0..shape.len], shape) and r.pooled_shape.dtype == dtype;
        if (if (pooled) mlx.mlx_array_set(&out, r.pooled) != 0 else dsv41_alloc_uninit(&out, shape.ptr, shape.len, dtype) != 0) {
            _ = mlx.mlx_array_free(out);
            return error.ScratchAlloc;
        }
        r.arr = out;
        @memcpy(r.shape[0..shape.len], shape);
        r.ndim = shape.len;
        r.dtype = dtype;
        r.bytes = n;
        ps().held_bytes += n;
        return r.arr;
    }
    /// Rows [row0, row0 + rows) along axis 0 as a writer's destination: a row-contiguous view of the buffer at its
    /// offset (a new handle the caller owns).
    pub fn slice(r: *const Region, s: mlx.mlx_stream, row0: usize, rows: usize) !A {
        if (r.arr.ctx == null or row0 + rows > @as(usize, @intCast(r.shape[0]))) return error.ScratchSlice;
        var start: [max_ndim]c_int = @splat(0);
        var stop: [max_ndim]c_int = undefined;
        const step: [max_ndim]c_int = @splat(1);
        @memcpy(stop[0..r.ndim], r.shape[0..r.ndim]);
        start[0] = @intCast(row0);
        stop[0] = @intCast(row0 + rows);
        var o = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_slice(&o, r.arr, &start, r.ndim, &stop, r.ndim, &step, r.ndim, s));
        return o;
    }
    /// A writer's carrier (its one-word output), kept as the latest.
    pub fn wrote(r: *Region, carrier_: A) !void {
        try hold(&r.carrier, carrier_);
    }
    /// The latest writer's carrier, for a reader of our own to bind (`dep`); none before the first write.
    pub fn latest(r: *const Region) A {
        return r.carrier;
    }
    /// An output of the last reader of the region's words (kept until the next one replaces it).
    pub fn released(r: *Region, token: A) !void {
        try hold(&r.release, token);
    }
    /// What the next writer binds as `after`: the last reader's output, else `fallback` (an input it binds anyway).
    pub fn after(r: *const Region, fallback: A) A {
        return if (r.release.ctx != null) r.release else fallback;
    }
    /// The region as a reader that binds no carrier takes it, a new handle the caller owns: an alias of its buffer that
    /// MLX holds as written in the compute pass after the latest writer's (the host's event wait over an event already
    /// at 1, the region and the carrier its tracked inputs).
    pub fn publish(r: *Region, s: mlx.mlx_stream) !A {
        if (comptime !shimmed) return error.ScratchUnwritten;
        if (r.arr.ctx == null or r.carrier.ctx == null) return error.ScratchUnwritten;
        const gated = [2]A{ r.arr, r.carrier };
        var outs = [2]A{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer _ = mlx.mlx_array_free(outs[1]);
        errdefer _ = mlx.mlx_array_free(outs[0]);
        if (dsv41ev_wait(&gated, 2, try readyEvent(), 1, null, 0, true, s, &outs) != 0) return error.ScratchPublish;
        return outs[0];
    }
    /// The bank pool: the pool's array for the region (`ensure` takes it), a reference kept until `dropPooled`.
    pub fn adoptPooled(r: *Region, arr: A, sh: Shape) !void {
        try hold(&r.pooled, arr);
        r.pooled_shape = sh;
    }
    pub fn dropPooled(r: *Region) void {
        drop(&r.pooled);
        r.pooled_shape = .{};
    }
    pub fn free(r: *Region) void {
        drop(&r.carrier);
        drop(&r.release);
        drop(&r.arr);
        ps().held_bytes -= r.bytes;
        r.bytes = 0;
        r.ndim = 0;
    }
};

fn dtypeBytes(dt: mlx.mlx_dtype) u64 {
    return switch (dt) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}
fn hold(slot: *A, x: A) !void {
    var held = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(held);
    try mlx.check(mlx.mlx_array_set(&held, x));
    drop(slot);
    slot.* = held;
}
fn drop(slot: *A) void {
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = none;
}
