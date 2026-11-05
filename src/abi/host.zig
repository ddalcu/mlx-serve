//! The `Host` table: everything process-wide a plugin reaches (the allocator, the log, the MLX error latch, the kernel
//! cache, levers, metrics, the memory ceiling), `HostGpu`, the C API below MLX's kernel API, and `HostAne`, the Neural
//! Engine. A plugin receives it
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

    /// The Neural Engine; read it through `aneOf`, which also answers null for a host older than this field.
    ane: ?*const HostAne = null,
};

/// `h.ane`, or null where the host has none or predates it (its `size` ends before the field).
pub fn aneOf(h: *const Host) ?*const HostAne {
    if (h.size < @offsetOf(Host, "ane") + @sizeOf(?*const HostAne)) return null;
    const t = h.ane orelse return null;
    return if (t.size >= @sizeOf(HostAne)) t else null;
}

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
    /// `out` (a fresh handle) becomes an evaluated [shape] array of `dtype` (mlx-c's `mlx_dtype` value) over `base`'s
    /// bytes from `offset`, an MTLBuffer of its own over the same pages: MLX orders its work apart from `base`'s.
    /// Refused unless `base` is evaluated at its buffer's start, `offset` is page-aligned and the range fits. It holds
    /// `base` and adds no active memory; MLX cannot see aliases overlap, so tensors alive at once take disjoint ranges.
    alias: *const fn (ctx: ?*anyopaque, base: Array, offset: u64, shape: [*]const i32, ndim: usize, dtype: i32, out: *Array) callconv(.c) Verdict,
    /// MLX's command-buffer split: the operations (`ops`) and MB of bound arrays (`mb`) a command buffer takes before
    /// it is committed, from each stream's next commit check (below 1 keeps the current value); the previous pair back.
    /// The words are the same at any split. Optional: read it only when `size` reaches it (`gpu_min_size`).
    command_buffer_limits: *const fn (ctx: ?*anyopaque, ops: i32, mb: i32, prev_ops: *i32, prev_mb: *i32) callconv(.c) Verdict,
};

/// The `HostGpu` size every plugin may rely on (through `alias`): members past it are optional, read only when the
/// table's `size` reaches them, so a plugin built with a newer kit still loads on a host without them, and a plugin
/// built before them loads on a host with them.
pub const gpu_min_size: usize = @offsetOf(HostGpu, "command_buffer_limits");

/// Whether the table `g` (a host's, of any size) reaches its member `name`.
pub fn gpuHas(g: *const HostGpu, comptime name: []const u8) bool {
    return g.size >= @offsetOf(HostGpu, name) + @sizeOf(@FieldType(HostGpu, name));
}

pub const ane_version: u32 = 1;

/// A compiled MIL program (`HostAne.program_create`), a request binding one procedure's surfaces, and an IOSurface.
pub const AneProgram = opaque {};
pub const AneRequest = opaque {};
pub const AneSurface = opaque {};
/// A bank under construction, the compiled bank (one program of typed procedures), and a shared I/O plane.
pub const AneBank = opaque {};
pub const AneBankProgram = opaque {};
pub const AnePlane = opaque {};

/// One input or output of a compiled procedure: a MIL tensor of rank r laid on the ANE's (batch, channel, depth,
/// height, width) from the right, the last axis packed, with the strides the compiled model reports (rows padded to 64
/// bytes). `dtype`: 0 fp16, 1 fp32, 2 int8, 3 uint8.
pub const AneIo = extern struct {
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
};

/// A surface and where it is mapped; zero-filled when made.
pub const AneSurfaceView = extern struct {
    surface: ?*AneSurface = null,
    base: ?[*]u8 = null,
    bytes: usize = 0,
};

/// The Neural Engine (`lib/ane`), one runtime for the whole process: its compile cache, its limit on resident compiled
/// models (about 121 on M5) and its dies are shared by every user, the host's own offloads included. Mechanism only:
/// which layers or ops go there is the caller's choice. Any thread; one caller at a time per program instance, bank
/// program or request, while different ones evaluate concurrently. Failures return null or a negative/zero result with
/// `err` filled (`err_len` bytes, NUL-terminated); the caller runs its GPU path instead.
pub const HostAne = extern struct {
    size: u32 = @sizeOf(HostAne),
    version: u32 = ane_version,
    ctx: ?*anyopaque,

    // ── Programs: any MIL ──
    /// One MIL program (`main`, or `procedureNNN` functions) over one weight blob, loaded `instances` times, load i
    /// pinned to die `dies[i]` (`dies` null, or 0 = no hint; two loads without hints can share a die). Takes `weights`
    /// (`malloc`'d, freed once compiled). The compile cache keys on the MIL text and the weights.
    program_create: *const fn (ctx: ?*anyopaque, name: [*:0]const u8, mil: [*:0]const u8, weights: ?*anyopaque, weight_bytes: usize, instances: u32, dies: ?[*]const u32, err: [*]u8, err_len: usize) callconv(.c) ?*AneProgram,
    program_free: *const fn (ctx: ?*anyopaque, p: ?*AneProgram) callconv(.c) void,
    /// The procedure index of the MIL function `function`; -1 when absent.
    program_procedure: *const fn (ctx: ?*anyopaque, p: *AneProgram, function: [*:0]const u8) callconv(.c) i32,
    program_io_count: *const fn (ctx: ?*anyopaque, p: *AneProgram, procedure: u32, output: bool) callconv(.c) u32,
    /// Inputs and outputs bind by name in the order this lists them; false for an index past the count.
    program_io: *const fn (ctx: ?*anyopaque, p: *AneProgram, procedure: u32, output: bool, index: u32, out: *AneIo) callconv(.c) bool,
    /// Load cost of the create: a fresh compile, or a cache hit.
    program_compile_seconds: *const fn (ctx: ?*anyopaque, p: *const AneProgram) callconv(.c) f64,
    program_cache_hit: *const fn (ctx: ?*anyopaque, p: *const AneProgram) callconv(.c) bool,
    /// An IOSurface of at least `bytes` (16 KB-aligned, host-mapped, zeroed); false when the OS refuses it.
    surface_create: *const fn (ctx: ?*anyopaque, bytes: usize, out: *AneSurfaceView) callconv(.c) bool,
    surface_release: *const fn (ctx: ?*anyopaque, s: ?*AneSurface) callconv(.c) void,
    /// Procedure `procedure` of load `instance` over these surfaces, in `program_io`'s order. The surfaces must
    /// outlive the request.
    request_create: *const fn (ctx: ?*anyopaque, p: *AneProgram, instance: u32, procedure: u32, inputs: [*]const ?*AneSurface, n_in: u32, outputs: [*]const ?*AneSurface, n_out: u32, err: [*]u8, err_len: usize) callconv(.c) ?*AneRequest,
    /// Runs the request to completion on the calling thread.
    request_eval: *const fn (ctx: ?*anyopaque, r: *AneRequest, err: [*]u8, err_len: usize) callconv(.c) bool,
    request_free: *const fn (ctx: ?*anyopaque, r: ?*AneRequest) callconv(.c) void,

    // ── Banks: typed procedures (int8 weights, one row of int8 and an f32 scale per output channel, fp16 I/O) ──
    /// A shared I/O plane of at least `bytes`: every program of one shape on one die can bind the same pair, since
    /// evaluations on a die run in turn. Concurrent dies need their own.
    plane_create: *const fn (ctx: ?*anyopaque, bytes: usize) callconv(.c) ?*AnePlane,
    plane_free: *const fn (ctx: ?*anyopaque, p: ?*AnePlane) callconv(.c) void,
    plane_base: *const fn (ctx: ?*anyopaque, p: *AnePlane) callconv(.c) ?[*]f16,
    bank_create: *const fn (ctx: ?*anyopaque) callconv(.c) ?*AneBank,
    /// Frees a bank not yet finished.
    bank_free: *const fn (ctx: ?*anyopaque, b: ?*AneBank) callconv(.c) void,
    /// Appends one SwiGLU (`down(silu(gate x) * up x)`, input `[hidden][rows]` channel-major) and returns its procedure
    /// index, or -1. `gate`/`up` are `[ffn, hidden]`, `down` `[hidden, ffn]`. `limit` > 0 clamps as DeepSeek V4's
    /// `swiglu_limit` does: gate <= limit, |up| <= limit; 0 clamps at fp16's range only.
    bank_add_swiglu: *const fn (ctx: ?*anyopaque, b: *AneBank, hidden: u32, ffn: u32, rows: u32, gate_q: [*]const i8, gate_s: [*]const f32, up_q: [*]const i8, up_s: [*]const f32, down_q: [*]const i8, down_s: [*]const f32, limit: f32, err: [*]u8, err_len: usize) callconv(.c) i32,
    /// Appends one GatedDeltaNet input projection: `qkv` `[qkv_out, hidden]` and `z` `[z_out, hidden]` as one fused
    /// conv whose output rows are qkv first. Returns the procedure index, or -1.
    bank_add_gdn: *const fn (ctx: ?*anyopaque, b: *AneBank, hidden: u32, qkv_out: u32, z_out: u32, rows: u32, qkv_q: [*]const i8, qkv_s: [*]const f32, z_q: [*]const i8, z_s: [*]const f32, err: [*]u8, err_len: usize) callconv(.c) i32,
    /// Compiles (or restores from the cache) and loads the bank as one program pinned to die `die` (0 = no hint).
    /// Consumes the bank either way. Null planes get a per-program allocation.
    bank_finish: *const fn (ctx: ?*anyopaque, b: *AneBank, name: [*:0]const u8, die: i32, input: ?*AnePlane, output: ?*AnePlane, err: [*]u8, err_len: usize) callconv(.c) ?*AneBankProgram,
    bank_program_free: *const fn (ctx: ?*anyopaque, m: ?*AneBankProgram) callconv(.c) void,
    bank_program_input: *const fn (ctx: ?*anyopaque, m: *AneBankProgram) callconv(.c) ?[*]f16,
    bank_program_output: *const fn (ctx: ?*anyopaque, m: *AneBankProgram) callconv(.c) ?[*]f16,
    /// Runs procedure `procedure` over the input plane into the output plane, on the calling thread.
    bank_program_eval: *const fn (ctx: ?*anyopaque, m: *AneBankProgram, procedure: u32, err: [*]u8, err_len: usize) callconv(.c) bool,
    bank_program_compile_seconds: *const fn (ctx: ?*anyopaque, m: *const AneBankProgram) callconv(.c) f64,
    bank_program_cache_hit: *const fn (ctx: ?*anyopaque, m: *const AneBankProgram) callconv(.c) bool,

    // ── The shared limits ──
    /// Compiled models resident in the process now (every user's, the host's included).
    resident: *const fn (ctx: ?*anyopaque) callconv(.c) u32,
};

test "abi host: aneOf answers null for a host that predates the field, or a table smaller than this one" {
    const t = std.testing;
    var ane: HostAne = undefined;
    ane.size = @sizeOf(HostAne);
    var h: Host = undefined;
    h.size = @sizeOf(Host);
    h.ane = &ane;
    try t.expect(aneOf(&h) == &ane);
    h.size = @offsetOf(Host, "ane");
    try t.expect(aneOf(&h) == null);
    h.size = @sizeOf(Host);
    ane.size = @sizeOf(HostAne) - 8;
    try t.expect(aneOf(&h) == null);
    h.ane = null;
    try t.expect(aneOf(&h) == null);
}

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
        .{ "ane", 136 },
    };
    inline for (host_fields) |f| try t.expectEqual(f[1], @offsetOf(Host, f[0]));
    try t.expectEqual(@as(usize, 144), @sizeOf(Host));
    try t.expectEqual(host_fields.len, @typeInfo(Host).@"struct".field_names.len);

    const gpu_fields = [_]struct { []const u8, usize }{
        .{ "size", 0 },              .{ "version", 4 },       .{ "ctx", 8 },
        .{ "event_metal", 16 },      .{ "event_host", 24 },   .{ "wait", 32 },
        .{ "signal", 40 },           .{ "event_set", 48 },    .{ "event_value", 56 },
        .{ "event_stats", 64 },      .{ "alloc_uninit", 72 }, .{ "launch", 80 },
        .{ "buffer_addresses", 88 }, .{ "alias", 96 },       .{ "command_buffer_limits", 104 },
    };
    inline for (gpu_fields) |f| try t.expectEqual(f[1], @offsetOf(HostGpu, f[0]));
    try t.expectEqual(@as(usize, 112), @sizeOf(HostGpu));
    try t.expectEqual(@as(usize, 104), gpu_min_size);
    try t.expectEqual(gpu_fields.len, @typeInfo(HostGpu).@"struct".field_names.len);

    const ane_fields = [_][]const u8{
        "size",                    "version",                      "ctx",                    "program_create",
        "program_free",            "program_procedure",            "program_io_count",       "program_io",
        "program_compile_seconds", "program_cache_hit",            "surface_create",         "surface_release",
        "request_create",          "request_eval",                 "request_free",           "plane_create",
        "plane_free",              "plane_base",                   "bank_create",            "bank_free",
        "bank_add_swiglu",         "bank_add_gdn",                 "bank_finish",            "bank_program_free",
        "bank_program_input",      "bank_program_output",          "bank_program_eval",      "bank_program_compile_seconds",
        "bank_program_cache_hit",  "resident",
    };
    inline for (ane_fields, 0..) |f, i| try t.expectEqual(if (i < 2) i * 4 else (i - 1) * 8, @offsetOf(HostAne, f));
    try t.expectEqual(@as(usize, 8 * (ane_fields.len - 1)), @sizeOf(HostAne));
    try t.expectEqual(ane_fields.len, @typeInfo(HostAne).@"struct".field_names.len);
    // `msv_ane_io`'s layout (lib/ane/ane_bridge.h).
    try t.expectEqual(@as(usize, 184), @sizeOf(AneIo));
    try t.expectEqual(@as(usize, 152), @offsetOf(AneIo, "batch_stride"));

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
