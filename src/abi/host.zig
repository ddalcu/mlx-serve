//! The `Host` table: everything process-wide a plugin reaches (the allocator, the log, the MLX error latch, the kernel
//! cache, levers, metrics, the memory ceiling) and `HostGpu`, the C API below MLX's kernel API. A plugin receives it
//! at load (`LoadCtx.host`) and keeps the pointer in its own per-load state: sdk holds no copy. Extern layouts, the C
//! calling convention, `size` first, grows at its end. Called only from the GPU thread, unless a member says otherwise.

const std = @import("std");

const common = @import("common.zig");
pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Array = common.Array;
pub const Stream = common.Stream;

pub const version: u32 = 1;

/// `log.Level`'s order.
pub const LogLevel = enum(u8) { err, warn, info, debug };

/// Which latch checkpoint a check is: a prefill chunk or a decode step (their fault injectors count apart).
pub const CheckSite = enum(u8) { prefill_chunk, decode_step };

/// A consumed latch: nothing, a memory-class failure (a 503), or any other MLX failure.
pub const Latched = enum(u8) { none, out_of_memory, failure };

pub const KernelBackend = enum(u8) { metal, cuda };

/// One custom kernel as mlx-c's `mlx_fast_{metal,cuda}_kernel_new` takes it. The cache key is
/// (backend, `name`, `digest`); `digest` names the source and header (the caller's hash, usually comptime).
pub const KernelDesc = extern struct {
    size: u32 = @sizeOf(KernelDesc),
    backend: KernelBackend = .metal,
    ensure_row_contiguous: bool = true,
    /// Metal only.
    atomic_outputs: bool = false,
    /// CUDA only: dynamic shared memory bytes.
    shared_memory: i32 = 0,
    digest: u64,
    name: Str,
    source: Str,
    header: Str = .{},
    inputs: Span(Str),
    outputs: Span(Str),
};

pub const Host = extern struct {
    size: u32 = @sizeOf(Host),
    version: u32 = version,
    ctx: ?*anyopaque,

    // The allocator (`std.mem.Allocator.VTable` minus the return address; `align_log2` is `std.mem.Alignment`).
    alloc: *const fn (ctx: ?*anyopaque, len: usize, align_log2: u8) callconv(.c) ?[*]u8,
    resize: *const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) bool,
    remap: *const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) ?[*]u8,
    free: *const fn (ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8) callconv(.c) void,

    // The log: ask `log_enabled` before formatting. Any thread.
    log_enabled: *const fn (ctx: ?*anyopaque, level: LogLevel) callconv(.c) bool,
    log: *const fn (ctx: ?*anyopaque, level: LogLevel, line: Str) callconv(.c) void,

    // The MLX error latch.
    /// An MLX call failed since the last consume (the `had_error` a best-effort op snapshots first).
    latch_pending: *const fn (ctx: ?*anyopaque) callconv(.c) bool,
    /// Consume the latch at a checkpoint.
    latch_check: *const fn (ctx: ?*anyopaque, site: CheckSite) callconv(.c) Latched,
    /// Drop a latch the op just raised unless one was pending before it (`had_error`).
    latch_drop_unless: *const fn (ctx: ?*anyopaque, had_error: bool) callconv(.c) void,
    /// A sticky CUDA fault: the process serves nothing until it restarts. Any thread.
    gpu_context_lost: *const fn (ctx: ?*anyopaque) callconv(.c) bool,

    /// The kernel cache: the process's one kernel object per (backend, name, digest), made on the first ask and kept
    /// for the process; null when the backend is not this build's or mlx-c refused. The handle is the mlx-c kernel's
    /// `ctx` (`mlx_fast_metal_kernel` / `mlx_fast_cuda_kernel`); a Metal one launches through `HostGpu.launch` too.
    kernel: *const fn (ctx: ?*anyopaque, desc: *const KernelDesc) callconv(.c) ?*anyopaque,

    /// A lever's raw value; `ptr` null = unset. Read once and keep it: a lookup scans the environment.
    lever: *const fn (ctx: ?*anyopaque, name: Str) callconv(.c) Str,

    /// A named counter's word, added to with a relaxed atomic add; null when metrics are off or the name is unknown,
    /// so a plugin resolves it once and pays nothing per add when off. Any thread.
    counter: *const fn (ctx: ?*anyopaque, name: Str) callconv(.c) ?*u64,

    /// Bytes the plugin may still allocate on the GPU right now: the host's ceiling less what is resident.
    memory_ceiling: *const fn (ctx: ?*anyopaque) callconv(.c) u64,

    gpu: *const HostGpu,
};

pub const gpu_version: u32 = 1;

pub const EventKind = enum(u8) { metal, host };

/// An event across rounds: `id` names it to `wait`/`signal`; `object` is what an off-thread signaller holds
/// (the id<MTLSharedEvent>, or the address of the host event's int64 word).
pub const Event = extern struct {
    id: i32 = 0,
    kind: EventKind = .metal,
    object: u64 = 0,
};

pub const EventStats = extern struct {
    size: u32 = @sizeOf(EventStats),
    gpu_waits: i64 = 0,
    gpu_signals: i64 = 0,
    /// Host waits already satisfied when evaluated.
    host_ready: i64 = 0,
    host_blocked: i64 = 0,
    host_timeouts: i64 = 0,
    host_wait_ns: i64 = 0,
    /// Waits on a CPU stream over a Metal event (pass-through).
    cpu_passthrough: i64 = 0,
    events: i64 = 0,
};

/// mlx-c handles the GPU calls pass through (by value, one context pointer each).
pub const VectorArray = extern struct { ctx: ?*anyopaque = null };
pub const KernelConfig = extern struct { ctx: ?*anyopaque = null };

/// The C API below MLX's kernel API: one implementation in the host against MLX's internals, re-verified on each MLX
/// bump. A graph without it (Linux, Windows) refuses every call `unsupported`.
pub const HostGpu = extern struct {
    size: u32 = @sizeOf(HostGpu),
    version: u32 = gpu_version,
    ctx: ?*anyopaque,

    /// An MTLSharedEvent on MLX's device at `start`, kept for the process.
    event_metal: *const fn (ctx: ?*anyopaque, start: u64, out: *Event) callconv(.c) Verdict,
    /// A host event over an 8-aligned int64 word, for CPU streams; a wait gives up after `timeout_ns`.
    event_host: *const fn (ctx: ?*anyopaque, word: *i64, timeout_ns: i64, out: *Event) callconv(.c) Verdict,
    /// `outs[i]` (fresh handles) alias `xs[i]` and are read only once `event` reaches `value` (>= 1). On a GPU stream
    /// the wait is encoded into the stream's current command buffer. The barrier choice: `deps` order the wait after
    /// them; `track_inputs` also orders it after the producers of `xs` (only for GPU-produced inputs).
    wait: *const fn (ctx: ?*anyopaque, xs: [*]const Array, n: usize, event: i32, value: u64, deps: ?[*]const Array, n_deps: usize, track_inputs: bool, stream: Stream, outs: [*]Array) callconv(.c) Verdict,
    /// `outs` alias `xs`; the GPU hands `value` to a Metal event after every pass encoded before it.
    signal: *const fn (ctx: ?*anyopaque, xs: [*]const Array, n: usize, event: i32, value: u64, stream: Stream, outs: [*]Array) callconv(.c) Verdict,
    /// Hand `value` to an event from any thread (not an MLX call): the read pool's completion.
    event_set: *const fn (ctx: ?*anyopaque, event: Event, value: u64) callconv(.c) void,
    /// The event's current value (0 for an unknown id). Any thread.
    event_value: *const fn (ctx: ?*anyopaque, event: i32) callconv(.c) u64,
    event_stats: *const fn (ctx: ?*anyopaque, out: *EventStats) callconv(.c) void,
    /// `out` (a fresh handle) becomes an evaluated [shape] array of `dtype` (mlx-c's `mlx_dtype` value) over one
    /// buffer from MLX's allocator, its bytes unspecified until written.
    alloc_uninit: *const fn (ctx: ?*anyopaque, shape: [*]const i32, ndim: usize, dtype: i32, out: *Array) callconv(.c) Verdict,
    /// `mlx_fast_metal_kernel_apply`'s contract; a kernel from the host's cache launches MLX's built pipeline directly
    /// once MLX has built it (no per-launch source regeneration). Bit-identical to MLX's apply.
    launch: *const fn (ctx: ?*anyopaque, outputs: *VectorArray, kernel: ?*anyopaque, inputs: VectorArray, config: KernelConfig, stream: Stream) callconv(.c) Verdict,
    /// Each evaluated, row-contiguous GPU array's buffer address + offset, for kernels reading sources from a table.
    buffer_addresses: *const fn (ctx: ?*anyopaque, arrays: [*]const Array, n: usize, out: [*]u64) callconv(.c) Verdict,
};

test "abi host: the table layouts are pinned (size first, offsets fixed, growth only at the end)" {
    const t = std.testing;
    if (@sizeOf(usize) != 8) return error.SkipZigTest;
    const host_fields = [_]struct { []const u8, usize }{
        .{ "size", 0 },              .{ "version", 4 },          .{ "ctx", 8 },
        .{ "alloc", 16 },            .{ "resize", 24 },          .{ "remap", 32 },
        .{ "free", 40 },             .{ "log_enabled", 48 },     .{ "log", 56 },
        .{ "latch_pending", 64 },    .{ "latch_check", 72 },     .{ "latch_drop_unless", 80 },
        .{ "gpu_context_lost", 88 }, .{ "kernel", 96 },          .{ "lever", 104 },
        .{ "counter", 112 },         .{ "memory_ceiling", 120 }, .{ "gpu", 128 },
    };
    inline for (host_fields) |f| try t.expectEqual(f[1], @offsetOf(Host, f[0]));
    try t.expectEqual(@as(usize, 136), @sizeOf(Host));
    try t.expectEqual(host_fields.len, @typeInfo(Host).@"struct".field_names.len);

    const gpu_fields = [_]struct { []const u8, usize }{
        .{ "size", 0 },              .{ "version", 4 },       .{ "ctx", 8 },
        .{ "event_metal", 16 },      .{ "event_host", 24 },   .{ "wait", 32 },
        .{ "signal", 40 },           .{ "event_set", 48 },    .{ "event_value", 56 },
        .{ "event_stats", 64 },      .{ "alloc_uninit", 72 }, .{ "launch", 80 },
        .{ "buffer_addresses", 88 },
    };
    inline for (gpu_fields) |f| try t.expectEqual(f[1], @offsetOf(HostGpu, f[0]));
    try t.expectEqual(@as(usize, 96), @sizeOf(HostGpu));
    try t.expectEqual(gpu_fields.len, @typeInfo(HostGpu).@"struct".field_names.len);

    // By-pointer structs open with `size`; by-value ones are frozen per version.
    try t.expectEqual(@as(usize, 0), @offsetOf(KernelDesc, "size"));
    try t.expectEqual(@as(usize, 16), @offsetOf(KernelDesc, "digest"));
    try t.expectEqual(@as(usize, 24), @offsetOf(KernelDesc, "name"));
    try t.expectEqual(@as(usize, 104), @sizeOf(KernelDesc));
    try t.expectEqual(@as(usize, 0), @offsetOf(EventStats, "size"));
    try t.expectEqual(@as(usize, 72), @sizeOf(EventStats));
    try t.expectEqual(@as(usize, 16), @sizeOf(Event));
    try t.expectEqual(@as(usize, 8), @offsetOf(Event, "object"));
    try t.expectEqual(@as(usize, 24), @sizeOf(Verdict));
}
