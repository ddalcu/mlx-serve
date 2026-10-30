//! The built-in `Host` table (`abi/host.zig`), filled from the process's real facilities: the allocator it is given,
//! `log.zig`, the MLX error latch in `mlx/ffi.zig`, a kernel cache it owns, the environment (until `src/env.zig`),
//! `--metrics`' counters and the memory ceiling the caller supplies. Built-ins and outside plugins get the same table.
//!
//! `HostGpu` delegates to the shims compiled with mlx-stream (`sdk/weights/streamed/csrc/`: `mlx_event_shim.cpp`,
//! `mlx_alloc_shim.cpp` with its `mlx_launch_cache.inc` and `mimo_gpu_address.inc`, `q3_event_shim.mm`): the one implementation over MLX's
//! internals until P4 moves it here. A graph without them (Linux, Windows) refuses every call `unsupported`.

const std = @import("std");
const env = @import("../env.zig");
const abi = @import("../abi/host.zig");
const mlx = @import("../mlx/ffi.zig");
const log = @import("../sdk/util/log.zig");
const metrics = @import("metrics/metrics.zig");

pub const Options = struct {
    allocator: std.mem.Allocator,
    /// `--metrics`; null = off (every `counter` is null).
    metrics: ?*metrics.Metrics = null,
    /// Bytes a plugin may still allocate on the GPU; the scheduler passes its admission view.
    ceiling: *const fn (ctx: ?*anyopaque) u64 = workingSetHeadroom,
    ceiling_ctx: ?*anyopaque = null,
};

/// Lives as long as any plugin holds its table: `ctx` points at it, so it is initialised in place.
pub const Builtin = struct {
    opts: Options,
    kernels: std.HashMapUnmanaged(KernelKey, *anyopaque, KernelKey.Ctx, 80) = .empty,
    table: abi.Host,

    pub fn init(self: *Builtin, opts: Options) void {
        self.* = .{ .opts = opts, .table = .{
            .ctx = self,
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
            .log_enabled = logEnabled,
            .log = logLine,
            .latch_pending = latchPending,
            .latch_check = latchCheck,
            .latch_drop_unless = latchDropUnless,
            .gpu_context_lost = gpuContextLost,
            .kernel = kernel,
            .lever = lever,
            .counter = counter,
            .memory_ceiling = memoryCeiling,
            .gpu = &gpu_table,
        } };
    }

    /// Frees the cached kernels: only when no plugin can launch them again (tests, process exit).
    pub fn deinit(self: *Builtin) void {
        var it = self.kernels.iterator();
        while (it.next()) |e| {
            freeKernel(e.key_ptr.backend, e.value_ptr.*);
            self.opts.allocator.free(e.key_ptr.name);
        }
        self.kernels.deinit(self.opts.allocator);
    }

    pub fn host(self: *const Builtin) *const abi.Host {
        return &self.table;
    }
};

/// The process's table: one kernel cache, latch and log for every plugin, built at the server's start and kept for the
/// process.
const Process = struct { builtin: Builtin = undefined, ready: bool = false };
pub const process_shared = @import("../sdk/util/shared.zig").Shared(Process, "plugin_host_table", .{});

/// The process's `Host` table, built on the first call and published (`host_table.zig`), where every model load reads
/// it for its plugin (`LoadCtx.host`). The host's plugin startup calls it, before any model loads.
pub fn processHost() *const abi.Host {
    const p = process_shared.get();
    if (!p.ready) {
        p.builtin.init(.{ .allocator = std.heap.c_allocator });
        p.ready = true;
    }
    @import("host_table.zig").published.get().host = p.builtin.host();
    return p.builtin.host();
}

fn self_(ctx: ?*anyopaque) *Builtin {
    return @ptrCast(@alignCast(ctx.?));
}

fn alignment(align_log2: u8) std.mem.Alignment {
    return @fromBackingInt(@intCast(@as(std.math.Log2Int(usize), @intCast(align_log2))));
}

fn alloc(ctx: ?*anyopaque, len: usize, align_log2: u8) callconv(.c) ?[*]u8 {
    return self_(ctx).opts.allocator.rawAlloc(len, alignment(align_log2), 0);
}
fn resize(ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) bool {
    return self_(ctx).opts.allocator.rawResize(ptr[0..len], alignment(align_log2), new_len, 0);
}
fn remap(ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8, new_len: usize) callconv(.c) ?[*]u8 {
    return self_(ctx).opts.allocator.rawRemap(ptr[0..len], alignment(align_log2), new_len, 0);
}
fn free(ctx: ?*anyopaque, ptr: [*]u8, len: usize, align_log2: u8) callconv(.c) void {
    self_(ctx).opts.allocator.rawFree(ptr[0..len], alignment(align_log2), 0);
}

fn level(l: abi.LogLevel) log.Level {
    return switch (l) {
        .err => .err,
        .warn => .warn,
        .info => .info,
        .debug => .debug,
    };
}
fn logEnabled(_: ?*anyopaque, l: abi.LogLevel) callconv(.c) bool {
    return log.enabled(level(l));
}
fn logLine(_: ?*anyopaque, l: abi.LogLevel, line: abi.Str) callconv(.c) void {
    log.atLevel(level(l), "{s}", .{line.slice()});
}

fn latchPending(_: ?*anyopaque) callconv(.c) bool {
    return mlx.errorPending();
}
fn latchCheck(_: ?*anyopaque, site: abi.CheckSite) callconv(.c) abi.Latched {
    const r = switch (site) {
        .prefill_chunk => mlx.checkError(),
        .decode_step => mlx.checkErrorDecode(),
    };
    r catch |e| return if (e == error.OutOfMemory) .out_of_memory else .failure;
    return .none;
}
fn latchDropUnless(_: ?*anyopaque, had_error: bool) callconv(.c) void {
    mlx.dropLatchedErrorUnless(had_error);
}
fn gpuContextLost(_: ?*anyopaque) callconv(.c) bool {
    return mlx.gpuContextLost();
}

const KernelKey = struct {
    backend: abi.KernelBackend,
    digest: u64,
    name: []const u8,

    const Ctx = struct {
        pub fn hash(_: Ctx, k: KernelKey) u64 {
            return std.hash.Wyhash.hash(k.digest +% @backingInt(k.backend), k.name);
        }
        pub fn eql(_: Ctx, a: KernelKey, b: KernelKey) bool {
            return a.backend == b.backend and a.digest == b.digest and std.mem.eql(u8, a.name, b.name);
        }
    };
};

fn kernel(ctx: ?*anyopaque, desc: *const abi.KernelDesc) callconv(.c) ?*anyopaque {
    const self = self_(ctx);
    const key: KernelKey = .{ .backend = desc.backend, .digest = desc.digest, .name = desc.name.slice() };
    if (self.kernels.get(key)) |h| return h;
    const h = makeKernel(self.opts.allocator, desc) catch null orelse return null;
    const gpa = self.opts.allocator;
    const name = gpa.dupe(u8, key.name) catch return keepUncached(desc.backend, h);
    self.kernels.put(gpa, .{ .backend = key.backend, .digest = key.digest, .name = name }, h) catch {
        gpa.free(name);
        return keepUncached(desc.backend, h);
    };
    return h;
}

/// Out of memory for the cache entry: free the kernel rather than hand out one nobody will free.
fn keepUncached(backend: abi.KernelBackend, h: *anyopaque) ?*anyopaque {
    freeKernel(backend, h);
    return null;
}

fn makeKernel(gpa: std.mem.Allocator, desc: *const abi.KernelDesc) !?*anyopaque {
    switch (desc.backend) {
        .metal => if (!mlx.metalKernelsAvailable()) return null,
        .cuda => if (!mlx.cudaAvailable()) return null,
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const name = try a.dupeSentinel(u8, desc.name.slice(), 0);
    const source = try a.dupeSentinel(u8, desc.source.slice(), 0);
    const header = try a.dupeSentinel(u8, desc.header.slice(), 0);
    const in_vec = try nameVector(a, desc.inputs.slice());
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = try nameVector(a, desc.outputs.slice());
    defer _ = mlx.mlx_vector_string_free(out_vec);
    return switch (desc.backend) {
        .metal => if (shims) |s|
            s.mlxlc_kernel_new(name, in_vec, out_vec, source, header, desc.ensure_row_contiguous, desc.atomic_outputs).ctx
        else
            mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, desc.ensure_row_contiguous, desc.atomic_outputs).ctx,
        .cuda => mlx.mlx_fast_cuda_kernel_new(name, in_vec, out_vec, source, header, desc.ensure_row_contiguous, desc.shared_memory).ctx,
    };
}

fn nameVector(a: std.mem.Allocator, names: []const abi.Str) !mlx.mlx_vector_string {
    const ptrs = try a.alloc([*:0]const u8, names.len);
    for (names, ptrs) |n, *p| p.* = try a.dupeSentinel(u8, n.slice(), 0);
    return mlx.mlx_vector_string_new_data(ptrs.ptr, ptrs.len);
}

fn freeKernel(backend: abi.KernelBackend, h: ?*anyopaque) void {
    if (h == null) return;
    switch (backend) {
        .metal => if (shims) |s| s.mlxlc_kernel_free(.{ .ctx = h }) else {
            _ = mlx.mlx_fast_metal_kernel_free(.{ .ctx = h });
        },
        .cuda => mlx.mlx_fast_cuda_kernel_free(.{ .ctx = h }),
    }
}

fn lever(_: ?*anyopaque, name: abi.Str) callconv(.c) abi.Str {
    var buf: [256]u8 = undefined;
    const n = name.slice();
    if (n.len >= buf.len) return .{};
    @memcpy(buf[0..n.len], n);
    buf[n.len] = 0;
    const v = env.byName(buf[0..n.len :0].ptr) orelse return .{};
    return abi.Str.of(std.mem.span(v));
}

fn counter(ctx: ?*anyopaque, name: abi.Str) callconv(.c) ?*u64 {
    const m = self_(ctx).opts.metrics orelse return null;
    const info = @typeInfo(metrics.Metrics).@"struct";
    inline for (info.field_names, info.field_types) |f, T| {
        if (T == metrics.Counter and std.mem.eql(u8, f, name.slice())) return &@field(m, f).value.raw;
    }
    return null;
}

fn memoryCeiling(ctx: ?*anyopaque) callconv(.c) u64 {
    const self = self_(ctx);
    return self.opts.ceiling(self.opts.ceiling_ctx);
}

/// The default ceiling: the device's working-set limit less MLX's active bytes.
fn workingSetHeadroom(_: ?*anyopaque) u64 {
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    return mlx.maxRecommendedWorkingSet() -| active;
}

// ── HostGpu ──

/// The mlx-stream shims, on the graphs that compile them.
const shims: ?type = if (@import("build_options").mlx_stream) struct {
    pub extern "c" fn dsv41ev_create_metal(start: u64, object: *u64) i32;
    pub extern "c" fn dsv41ev_create_host(word: *i64, timeout_ns: i64) i32;
    pub extern "c" fn dsv41ev_wait(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, deps: ?[*]const mlx.mlx_array, n_deps: usize, track_inputs: bool, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int;
    pub extern "c" fn dsv41ev_signal(xs: [*]const mlx.mlx_array, n: usize, event: i32, value: u64, s: mlx.mlx_stream, outs: [*]mlx.mlx_array) c_int;
    pub extern "c" fn dsv41ev_value(event: i32) u64;
    pub extern "c" fn dsv41ev_stats(out: *[8]i64) void;
    pub extern "c" fn dsv41ev_last_error() [*:0]const u8;
    pub extern "c" fn q3ld_mtl_signal(event: ?*anyopaque, value: u64) void;
    pub extern "c" fn dsv41_alloc_uninit(out: *mlx.mlx_array, shape: [*]const c_int, ndim: usize, dtype: mlx.mlx_dtype) c_int;
    pub extern "c" fn dsv41_alloc_last_error() [*:0]const u8;
    pub extern "c" fn mlxlc_kernel_new(name: [*:0]const u8, input_names: mlx.mlx_vector_string, output_names: mlx.mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx.mlx_fast_metal_kernel;
    pub extern "c" fn mlxlc_kernel_free(k: mlx.mlx_fast_metal_kernel) void;
    pub extern "c" fn mlxlc_apply(outputs: *mlx.mlx_vector_array, k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, config: mlx.mlx_fast_metal_kernel_config, stream: mlx.mlx_stream) c_int;
    pub extern "c" fn mimo_gpu_addresses(arrs: [*]const mlx.mlx_array, n: usize, out: [*]u64) c_int;
} else null;

pub const gpu_table: abi.HostGpu = .{
    .ctx = null,
    .event_metal = eventMetal,
    .event_host = eventHost,
    .wait = wait,
    .signal = signal,
    .event_set = eventSet,
    .event_value = eventValue,
    .event_stats = eventStats,
    .alloc_uninit = allocUninit,
    .launch = launch,
    .buffer_addresses = bufferAddresses,
};

const unsupported = abi.Verdict.refuse(.unsupported, "HostGpu is not built on this graph");

/// A shim's refusal: its reason goes to the log, the caller gets the call's name.
fn refused(comptime op: []const u8, why: [*:0]const u8) abi.Verdict {
    log.warn("[gpu] " ++ op ++ ": {s}\n", .{std.mem.span(why)});
    return abi.Verdict.refuse(.failed, op ++ " refused");
}

fn arrays(xs: [*]const abi.Array) [*]const mlx.mlx_array {
    return @ptrCast(xs);
}
fn stream(s: abi.Stream) mlx.mlx_stream {
    return .{ .ctx = s.ctx };
}

fn eventMetal(_: ?*anyopaque, start: u64, out: *abi.Event) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    var object: u64 = 0;
    const id = s.dsv41ev_create_metal(start, &object);
    if (id <= 0) return refused("event_metal", s.dsv41ev_last_error());
    out.* = .{ .id = id, .kind = .metal, .object = object };
    return .ok;
}
fn eventHost(_: ?*anyopaque, word: *i64, timeout_ns: i64, out: *abi.Event) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    const id = s.dsv41ev_create_host(word, timeout_ns);
    if (id <= 0) return refused("event_host", s.dsv41ev_last_error());
    out.* = .{ .id = id, .kind = .host, .object = @intFromPtr(word) };
    return .ok;
}
fn wait(_: ?*anyopaque, xs: [*]const abi.Array, n: usize, event: i32, value: u64, deps: ?[*]const abi.Array, n_deps: usize, track_inputs: bool, s_: abi.Stream, outs: [*]abi.Array) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    const d: ?[*]const mlx.mlx_array = if (deps) |p| arrays(p) else null;
    if (s.dsv41ev_wait(arrays(xs), n, event, value, d, n_deps, track_inputs, stream(s_), @ptrCast(outs)) != 0) return refused("wait", s.dsv41ev_last_error());
    return .ok;
}
fn signal(_: ?*anyopaque, xs: [*]const abi.Array, n: usize, event: i32, value: u64, s_: abi.Stream, outs: [*]abi.Array) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    if (s.dsv41ev_signal(arrays(xs), n, event, value, stream(s_), @ptrCast(outs)) != 0) return refused("signal", s.dsv41ev_last_error());
    return .ok;
}
fn eventSet(_: ?*anyopaque, event: abi.Event, value: u64) callconv(.c) void {
    const s = shims orelse return;
    switch (event.kind) {
        .metal => s.q3ld_mtl_signal(@ptrFromInt(event.object), value),
        .host => @atomicStore(i64, @as(*i64, @ptrFromInt(event.object)), @bitCast(value), .release),
    }
}
fn eventValue(_: ?*anyopaque, event: i32) callconv(.c) u64 {
    const s = shims orelse return 0;
    return s.dsv41ev_value(event);
}
fn eventStats(_: ?*anyopaque, out: *abi.EventStats) callconv(.c) void {
    const s = shims orelse {
        out.* = .{};
        return;
    };
    var v: [8]i64 = undefined;
    s.dsv41ev_stats(&v);
    out.* = .{ .gpu_waits = v[0], .gpu_signals = v[1], .host_ready = v[2], .host_blocked = v[3], .host_timeouts = v[4], .host_wait_ns = v[5], .cpu_passthrough = v[6], .events = v[7] };
}
fn allocUninit(_: ?*anyopaque, shape: [*]const i32, ndim: usize, dtype: i32, out: *abi.Array) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    const dt = std.enums.fromInt(mlx.mlx_dtype, dtype) orelse return abi.Verdict.refuse(.failed, "alloc_uninit: unknown dtype");
    if (s.dsv41_alloc_uninit(@ptrCast(out), shape, ndim, dt) != 0) return refused("alloc_uninit", s.dsv41_alloc_last_error());
    return .ok;
}
fn launch(_: ?*anyopaque, outputs: *abi.VectorArray, k: ?*anyopaque, inputs: abi.VectorArray, config: abi.KernelConfig, s_: abi.Stream) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    // A failure inside MLX's apply is on the latch; the shim's own lands in its last error.
    if (s.mlxlc_apply(@ptrCast(outputs), .{ .ctx = k }, .{ .ctx = inputs.ctx }, .{ .ctx = config.ctx }, stream(s_)) != 0) return refused("launch", s.dsv41_alloc_last_error());
    return .ok;
}
fn bufferAddresses(_: ?*anyopaque, arrs: [*]const abi.Array, n: usize, out: [*]u64) callconv(.c) abi.Verdict {
    const s = shims orelse return unsupported;
    if (s.mimo_gpu_addresses(arrays(arrs), n, out) != 0) return refused("buffer_addresses", s.dsv41_alloc_last_error());
    return .ok;
}

// ── Tests (CPU: no array is evaluated, no Metal device is made) ──

const testing = std.testing;
const sdk = @import("../sdk/host.zig");

test {
    _ = abi;
    _ = sdk;
    _ = @import("../sdk/testing/fake_host.zig");
    _ = @import("no_globals.zig");
}

fn fixedCeiling(ctx: ?*anyopaque) u64 {
    return @as(*const u64, @ptrCast(@alignCast(ctx.?))).*;
}

test "process host: built once and published where the engine's loads read it" {
    const host_table = @import("host_table.zig");
    const saved = host_table.published.get().*;
    defer host_table.published.get().* = saved;
    host_table.published.get().* = .{};
    try testing.expect(host_table.get() == null);
    const h = processHost();
    try testing.expectEqual(h, host_table.get().?);
    try testing.expectEqual(h, processHost());
    try testing.expectEqual(@as(u32, @sizeOf(abi.Host)), h.size);
}

test "builtin host: the allocator round-trips through the table (std.mem.Allocator over it)" {
    var b: Builtin = undefined;
    b.init(.{ .allocator = testing.allocator });
    defer b.deinit();
    const h = try sdk.Host.of(b.host());
    const a = h.allocator();
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(a);
    for (0..1000) |i| try list.append(a, @intCast(i));
    try testing.expectEqual(@as(u32, 999), list.items[999]);
    const p = try a.alignedAlloc(u8, .@"64", 100);
    try testing.expect(std.mem.isAligned(@intFromPtr(p.ptr), 64));
    a.free(p);
    // testing.allocator reports a leak or a mismatched free at the end of the test
}

test "builtin host: a log line reaches the log sink at its level" {
    var b: Builtin = undefined;
    b.init(.{ .allocator = testing.allocator });
    defer b.deinit();
    const h = try sdk.Host.of(b.host());

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [512]u8 = undefined;
    var path_buf: [600]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/h.log", .{root[0..try tmp.dir.realPath(testing.io, &root)]}, 0);
    try log.openFile(path, 0);
    h.log(.info, "plugin says {d}\n", .{42});
    h.log(.debug, "below the level\n", .{});
    log.closeFile();

    const text = try tmp.dir.readFileAlloc(testing.io, "h.log", testing.allocator, .limited(4096));
    defer testing.allocator.free(text);
    try testing.expect(std.mem.indexOf(u8, text, "plugin says 42\n") != null);
    try testing.expect(std.mem.indexOf(u8, text, "below the level") == null);
    try testing.expect(h.logEnabled(.err));
    try testing.expect(!h.logEnabled(.debug));
}

test "builtin host: the MLX error latch round-trips (pending, consume by class, drop-unless, context lost)" {
    var b: Builtin = undefined;
    b.init(.{ .allocator = testing.allocator });
    defer b.deinit();
    const h = try sdk.Host.of(b.host());
    mlx.resetFaultChunkForTest();
    defer mlx.resetGpuContextLostForTest();

    try testing.expect(!h.latchPending());
    try h.check(.prefill_chunk);

    mlx.latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try testing.expect(h.latchPending());
    try testing.expectError(error.OutOfMemory, h.check(.decode_step));
    try testing.expect(!h.latchPending());

    mlx.latchErrorForTest("[matmul] Last dimension of first input must match.");
    try testing.expectError(error.MlxFailure, h.check(.prefill_chunk));

    // A best-effort op's own latch is dropped; one pending before it belongs to someone else.
    mlx.latchErrorForTest("[slice_update] Invalid slice sizes.");
    h.dropLatchedErrorUnless(true);
    try testing.expect(h.latchPending());
    h.dropLatchedErrorUnless(false);
    try testing.expect(!h.latchPending());

    try testing.expect(!h.contextLost());
    mlx.latchErrorForTest("cudaGraphLaunch failed: an illegal memory access was encountered");
    try testing.expect(h.contextLost());
    mlx.resetFaultChunkForTest();
}

test "builtin host: metrics are no-ops when off and reach the named counter when on" {
    var off: Builtin = undefined;
    off.init(.{ .allocator = testing.allocator });
    defer off.deinit();
    const h_off = try sdk.Host.of(off.host());
    const c_off = h_off.counter("generation_tokens_total");
    try testing.expect(c_off.word == null);
    c_off.add(5); // nothing to write

    var m = metrics.Metrics.init();
    var on: Builtin = undefined;
    on.init(.{ .allocator = testing.allocator, .metrics = &m });
    defer on.deinit();
    const h = try sdk.Host.of(on.host());
    const c = h.counter("generation_tokens_total");
    c.add(3);
    c.add(4);
    try testing.expectEqual(@as(u64, 7), m.generation_tokens_total.load());
    try testing.expect(h.counter("no_such_counter").word == null);
    try testing.expect(h.counter("requests_running").word == null); // a gauge, not a counter
}

test "builtin host: levers read the environment (diagnostic semantics: unset or 0 is off), memory ceiling from the caller" {
    var ceiling: u64 = 12345;
    var b: Builtin = undefined;
    b.init(.{ .allocator = testing.allocator, .ceiling = fixedCeiling, .ceiling_ctx = &ceiling });
    defer b.deinit();
    const h = try sdk.Host.of(b.host());
    try testing.expectEqual(@as(?[]const u8, null), h.leverRaw("MLX_SERVE_HOST_TABLE_TEST_SURELY_UNSET"));
    try testing.expect(!h.leverOn("MLX_SERVE_HOST_TABLE_TEST_SURELY_UNSET"));
    const home = std.mem.span(env.get(.HOME) orelse return error.SkipZigTest);
    try testing.expectEqualStrings(home, h.leverRaw("HOME").?);
    try testing.expectEqual(@as(u64, 12345), h.memoryCeiling());
    ceiling = 7;
    try testing.expectEqual(@as(u64, 7), h.memoryCeiling());
}

test "builtin host: the kernel cache makes one kernel per (backend, name, digest)" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    var b: Builtin = undefined;
    b.init(.{ .allocator = testing.allocator });
    defer b.deinit();
    const h = try sdk.Host.of(b.host());
    const src = "uint i = thread_position_in_grid.x; out[i] = inp[i] + 1.0f;";
    const ins = [_]abi.Str{abi.Str.of("inp")};
    const outs = [_]abi.Str{abi.Str.of("out")};
    var desc: abi.KernelDesc = .{ .digest = sdk.digest(src, ""), .name = abi.Str.of("host_table_add1"), .source = abi.Str.of(src), .inputs = .of(&ins), .outputs = .of(&outs) };
    const k1 = h.kernel(&desc) orelse return error.NoKernel;
    try testing.expectEqual(k1, h.kernel(&desc).?);
    desc.digest +%= 1; // another source under the same name is another kernel
    const k2 = h.kernel(&desc) orelse return error.NoKernel;
    try testing.expect(k1 != k2);
    try testing.expectEqual(@as(u32, 2), b.kernels.count());
    desc.backend = .cuda; // not this build's backend
    try testing.expect(h.kernel(&desc) == null);
}

test "builtin host gpu: a host event round-trips through event_set and event_value; other graphs refuse by name" {
    const g = gpu_table;
    var word: i64 = 0;
    var ev: abi.Event = .{};
    const v = g.event_host(null, &word, std.time.ns_per_s, &ev);
    if (shims == null) {
        try testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
        return;
    }
    try testing.expect(v.isOk());
    try testing.expect(ev.id >= 1 and ev.kind == .host and ev.object == @intFromPtr(&word));
    g.event_set(null, ev, 9);
    try testing.expectEqual(@as(i64, 9), word);
    try testing.expectEqual(@as(u64, 9), g.event_value(null, ev.id));
    var st: abi.EventStats = .{};
    g.event_stats(null, &st);
    try testing.expect(st.events >= 1);
    // A refusal names the call; the shim's reason goes to the log.
    try testing.expectEqual(abi.Verdict.Code.failed, g.event_host(null, &word, 0, &ev).code);
}
