//! MLX side of the streamer's event gate (lib/expert_io/mlx_event_shim.cpp).
//! A gated wave reads slot arrays through `wait` aliases: on a GPU stream the
//! command buffer waits for the pool to hand the event the gate's value, so the
//! waves are committed before their bytes land; on a CPU stream a host event
//! holds the evaluating thread. Inference thread only, like every MLX call.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("sdk").mlx;

const c = if (builtin.os.tag == .macos) struct {
    pub extern fn dsv41ev_abi() i32;
    pub extern fn dsv41ev_create_metal(start: u64, object: *u64) i32;
    pub extern fn dsv41ev_create_host(word: *i64, timeout_ns: i64) i32;
    pub extern fn dsv41ev_wait(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, deps: ?[*]const mlx.mlx_array, n_deps: usize, track_inputs: bool, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int;
    pub extern fn dsv41ev_signal(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int;
    pub extern fn dsv41ev_value(event: i32) u64;
    pub extern fn dsv41ev_stats(out: *[8]i64) void;
    pub extern fn dsv41ev_last_error() [*:0]const u8;
} else @import("io_stub.zig").ev;

/// Test builds only: the shim's C ABI, for its own tests. Served code goes through the functions below.
pub const test_abi = if (builtin.is_test) struct {
    pub const abi = c;
} else struct {};

pub const abi_version = 2026092801;

/// An event the pool signals: `id` names it to `wait`, `object` is what the
/// pool's event class is armed with (the id<MTLSharedEvent>, or the word).
pub const Event = struct { id: i32, object: u64 };

/// An MTLSharedEvent on MLX's GPU device, at value 0. Creates the Metal device.
pub fn createMetal() !Event {
    var object: u64 = 0;
    const id = c.dsv41ev_create_metal(0, &object);
    if (id <= 0) return error.EventUnavailable;
    return .{ .id = id, .object = object };
}

/// A host event over the stream's word (Stream.eventWord), for CPU streams.
pub fn createHost(word: *i64, timeout_ns: i64) !Event {
    const id = c.dsv41ev_create_host(word, timeout_ns);
    if (id <= 0) return error.EventUnavailable;
    return .{ .id = id, .object = @intFromPtr(word) };
}

/// `outs[i]` alias `xs[i]`; nothing reads them before the event reaches
/// `value`. `deps` only order the wait (after them); `track_inputs` orders it
/// after the producers of `xs` (only for GPU-produced inputs). `outs` are fresh
/// handles (`mlx_array_new`): the shim assigns each alias into its handle.
pub fn wait(xs: []const mlx.mlx_array, event: Event, value: u64, deps: []const mlx.mlx_array, track_inputs: bool, stream: mlx.mlx_stream, outs: []mlx.mlx_array) !void {
    std.debug.assert(xs.len == outs.len and xs.len > 0);
    if (std.debug.runtime_safety) std.debug.assert(allFresh(outs));
    if (c.dsv41ev_wait(xs.ptr, xs.len, event.id, value, deps.ptr, deps.len, track_inputs, stream, outs.ptr) != 0) return error.EventWaitRefused;
}

/// `outs` alias `xs` (fresh handles, as `wait`'s); the GPU hands `value` to a
/// metal event after every pass encoded before it (probes).
pub fn signal(xs: []const mlx.mlx_array, event: Event, value: u64, stream: mlx.mlx_stream, outs: []mlx.mlx_array) !void {
    if (std.debug.runtime_safety) std.debug.assert(allFresh(outs));
    if (c.dsv41ev_signal(xs.ptr, xs.len, event.id, value, stream, outs.ptr) != 0) return error.EventSignalRefused;
}

/// Handles the shim may assign into: none holds an array (an `undefined` one is stack garbage, which the
/// shim's move-assign would dereference).
pub fn allFresh(outs: []const mlx.mlx_array) bool {
    for (outs) |o| if (o.ctx != null) return false;
    return true;
}

pub fn signaledValue(event: Event) u64 {
    return c.dsv41ev_value(event.id);
}

pub const Stats = struct { gpu_waits: i64, gpu_signals: i64, host_ready: i64, host_blocked: i64, host_timeouts: i64, host_wait_ns: i64, cpu_passthrough: i64, events: i64 };

pub fn stats() Stats {
    var w: [8]i64 = undefined;
    c.dsv41ev_stats(&w);
    return .{ .gpu_waits = w[0], .gpu_signals = w[1], .host_ready = w[2], .host_blocked = w[3], .host_timeouts = w[4], .host_wait_ns = w[5], .cpu_passthrough = w[6], .events = w[7] };
}

pub fn lastError() []const u8 {
    return std.mem.span(c.dsv41ev_last_error());
}
