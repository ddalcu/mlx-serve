//! Hot-cache budgets, residency publishing, SSD flush and shutdown persistence.

const std = @import("std");
const model_mod = @import("../../../sdk/config/model.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const metrics_mod = @import("../../../host/metrics/metrics.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const log = @import("../../../sdk/util/log.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const plugin_driver = @import("../plugin/driver.zig");
const ModelRegistry = model_registry_mod.ModelRegistry;
const LoadedModel = model_registry_mod.LoadedModel;
const testing = std.testing;

const Scheduler = @import("../scheduler.zig").Scheduler;
const Slot = @import("../slot.zig").Slot;
const drainSideQueues = @import("../runtime/loop.zig").drainSideQueues;

/// Republish the hot cache's residency for the connection thread's admission guard.
/// Called from the inference thread after a commit, eviction, invalidation or model switch.
pub const BUDGET_REVISE_WINDOW_NS: u64 = 10 * std.time.ns_per_s;

/// How a budget resolve differs from the load-time one: the cache's own resident bytes
/// are excluded from the machine read, and the resolver keeps quiet (`setBudget` logs).
pub const BudgetRevise = @import("../../../host/gpu_thread/scheduler.zig").BudgetRevise;

/// Re-clamp every resident model's hot-cache budget after residency changed (#364): the
/// load-time clamp read the machine with the other models on it and was never revisited, so
/// a model loaded beside a large one kept a ~0 budget for life. Each cache's own resident
/// entries are excluded from the read so a full cache cannot ratchet itself down.
/// `grow_only`: the free figure lags an unload's freed pages, so a revise that follows one may
/// raise a budget but never lower it; lowering evicts entries the settled figure would keep.
pub fn reviseHotCacheBudgets(sch: *Scheduler, grow_only: bool) void {
    const resolve = sch.prefix_cache_mem_resolver orelse return;
    sch.registry.mutex.lockUncancelable(sch.io);
    // The resolver publishes the process-global budget the admission guard reads, so the
    // current model goes last.
    for ([_]bool{ false, true }) |current_pass| {
        var it = sch.registry.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if ((entry == sch.current_model) != current_pass) continue;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |h| h else continue;
            if (!hc.ram_enabled) continue;
            const config = entry.config orelse continue;
            var idle: u64 = 0;
            const budget = resolve(config, sch.prefix_cache_mem_bytes, .{ .exclude_bytes = hc.residentBytes(), .quiet = true }, &idle);
            if (grow_only and prefix_cache_mod.HotPrefixCache.budgetShrinks(hc.max_kv_bytes, budget)) continue;
            hc.setBudget(budget);
            hc.ssd_idle_mem = idle;
        }
    }
    sch.registry.mutex.unlock(sch.io);
    publishHotCacheResidency(sch);
}

/// Active, parked and pending-capture state, counted once per model. Cache-owned snapshots bill separately.
/// Inference thread only; the caller must not hold registry.mutex.
pub fn moduleStateBytes(sch: *Scheduler) u64 {
    var bytes: u64 = 0;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        if (entry.state != .ready) continue;
        const t = entry.transformer orelse continue;
        bytes += plugin_driver.stateBytes(t);
    }
    return bytes;
}

/// The view must not outlive its cache: before `leaving`'s cache is freed, hand the view to
/// another resident chat model's, the default's when it has one, or to none.
pub fn moveHotCacheViewOff(sch: *Scheduler, leaving: *LoadedModel) void {
    const view = sch.hot_prefix_cache orelse return;
    if (leaving.prefix_cache == null or view != leaving.prefix_cache.?) return;
    sch.hot_prefix_cache = null;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        if (entry == leaving or entry.state != .ready or !ModelRegistry.chatCapable(entry)) continue;
        const hc = if (entry.prefix_cache) |h| h else continue;
        sch.hot_prefix_cache = hc;
        if (std.mem.eql(u8, sch.registry.default_id, entry.id)) break;
    }
}

pub fn publishHotCacheResidency(sch: *Scheduler) void {
    const bytes: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.resident_hot_cache_bytes.store(bytes, .monotonic);
    const reclaimable: u64 = if (sch.hot_prefix_cache) |hc| hc.reclaimableBytes() else 0;
    sch.reclaimable_hot_cache_bytes.store(reclaimable, .monotonic);
    // Live module state is published with the live KV (`publishLiveKvResidency`, per pass and per chunk), never here:
    // it is not evictable, and republishing live sessions here would drop the slot being prefilled.
    publishHotCacheDigests(sch);
    if (sch.metrics != null) publishCachedSessions(sch);
}

/// Caller must not hold `registry.mutex`.
fn publishCachedSessions(sch: *Scheduler) void {
    var rows: [metrics_mod.MAX_SESSIONS]metrics_mod.Session = undefined;
    var n: usize = 0;
    {
        sch.registry.mutex.lockUncancelable(sch.io);
        defer sch.registry.mutex.unlock(sch.io);
        var it = sch.registry.entries.valueIterator();
        outer: while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.state != .ready) continue;
            const hc = if (entry.prefix_cache) |h| h else continue;
            for (hc.entries.items) |*e| {
                if (n == rows.len) break :outer;
                const len: u32 = @intCast(@min(e.tokens.len, std.math.maxInt(u32)));
                rows[n] = .init(entry.id, .cached, len, len, 0, e.kv_bytes);
                rows[n].entry_id = e.id;
                n += 1;
            }
        }
    }
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    @memcpy(sch.cached_sessions[0..n], rows[0..n]);
    sch.cached_session_count = n;
}

/// Swap in a fresh digest snapshot and free the one it supersedes (inference thread only).
/// An allocation failure keeps the previous snapshot: a stale digest is a hint, an empty one
/// credits bytes that exist.
fn publishHotCacheDigests(sch: *Scheduler) void {
    const fresh: []prefix_cache_mod.HotPrefixCache.EntryDigest = if (sch.hot_prefix_cache) |hc|
        (hc.digestsAlloc(sch.allocator) catch return)
    else
        &.{};
    const residency: u64 = if (sch.hot_prefix_cache) |hc| hc.residentBytes() else 0;
    sch.digest_mu.lockUncancelable(sch.io);
    const old = sch.hot_cache_digests;
    sch.hot_cache_digests = fresh;
    // Published under the same lock as the digests it describes.
    sch.digest_residency = residency;
    sch.digest_mu.unlock(sch.io);
    if (old.len > 0) sch.allocator.free(old);
}

/// Connection-thread entry point: what an eviction pass can prove it will get back for this prompt.
pub fn reclaimableHotCacheBytesFor(sch: *Scheduler, prompt_tokens: []const u32) u64 {
    const fp = prefix_cache_mod.HotPrefixCache.prefixFingerprint(prompt_tokens);
    sch.digest_mu.lockUncancelable(sch.io);
    defer sch.digest_mu.unlock(sch.io);
    return prefix_cache_mod.HotPrefixCache.reclaimableFromDigests(
        sch.hot_cache_digests,
        sch.digest_residency,
        fp,
    );
}

/// One bounded SSD slice per resident model that still owes one (its oldest owed session first);
/// returns progress, busy, or done. The caller re-checks for work between calls.
/// Inference thread only: unloads run here too, so the collected pointers stay valid.
pub fn flushOwedDiskSlices(sch: *Scheduler) prefix_cache_mod.FlushResult {
    var owed: [16]prefix_cache_mod.FlushTarget = undefined;
    var result: prefix_cache_mod.FlushResult = .done;
    for (owed[0..diskCaches(sch, &owed, true)]) |o| {
        const slice = if (admitModuleStaging(sch, o.hc)) o.hc.flushOwedSlice(o.s) else .busy;
        if (slice == .progress) result = .progress;
        if (slice == .busy and result == .done) result = .busy;
    }
    return result;
}

/// Staged CPU bytes share the unified-memory ceiling with every resident binding's tensors.
pub fn stagedHostBytes(sch: *Scheduler) u64 {
    var bytes: u64 = 0;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        if (entry.state != .ready) continue;
        if (entry.prefix_cache) |hc| bytes +|= hc.stagedHostBytes();
    }
    return bytes;
}

const ModuleStagingProbe = struct {
    sch: *Scheduler,
    hc: *prefix_cache_mod.HotPrefixCache,
    cfg: *const model_mod.ModelConfig,

    fn fits(ctx: ?*anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(ctx.?));
        const need = self.hc.nextModuleStagingBytes();
        if (need == 0) return true;
        return need <= @import("../admission/prefill_chunk.zig").prefillHeadroomNow(self.cfg, stagedHostBytes(self.sch));
    }
};

fn admitModuleStaging(sch: *Scheduler, hc: *prefix_cache_mod.HotPrefixCache) bool {
    if (hc.nextModuleStagingBytes() == 0) return true;
    const cfg = cacheConfig(sch, hc) orelse return false;
    var probe: ModuleStagingProbe = .{ .sch = sch, .hc = hc, .cfg = cfg };
    const report = hc.evictLruToAdmit(0, &probe, ModuleStagingProbe.fits, false);
    if (report.entries > 0) publishHotCacheResidency(sch);
    return report.admitted;
}

fn cacheConfig(sch: *Scheduler, hc: *prefix_cache_mod.HotPrefixCache) ?*const model_mod.ModelConfig {
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        const entry = entry_ptr.*;
        if (entry.state == .ready and entry.prefix_cache == hc) return entry.config;
    }
    return null;
}

/// Neural Engine requests run on their workers while this thread parks, and still keep the Mac awake.
pub fn holdsSleepWhileParked(sch: *const Scheduler) bool {
    return sch.offloaded.items.len > 0;
}

/// The hot caches of ready models with a disk tier (only those owing a slice when `owed_only`).
/// Inference thread only: unloads run here too, so the collected pointers stay valid.
fn diskCaches(sch: *Scheduler, out: []prefix_cache_mod.FlushTarget, owed_only: bool) usize {
    var n: usize = 0;
    sch.registry.mutex.lockUncancelable(sch.io);
    defer sch.registry.mutex.unlock(sch.io);
    var it = sch.registry.entries.valueIterator();
    while (it.next()) |entry_ptr| {
        if (n == out.len) break;
        const entry = entry_ptr.*;
        if (entry.state != .ready) continue;
        const hc = if (entry.prefix_cache) |h| h else continue;
        if (hc.disk == null or (owed_only and !hc.disk_dirty)) continue;
        const xf = entry.transformer orelse continue;
        out[n] = .{ .hc = hc, .s = xf.s };
        n += 1;
    }
    return n;
}

/// SIGTERM's bounded SSD flush (`--shutdown-persist-secs`), on this thread once the loop has
/// stopped: what the idle flush has not landed, until the deadline `serve` armed. Off when unarmed.
pub fn persistAtShutdown(sch: *Scheduler) void {
    const deadline_ms = sch.shutdown_deadline_ms.load(.acquire);
    if (deadline_ms == 0) return;
    // A slot cancelled as `shutdown` flipped can still sit in the cleanup queue: commit it and
    // return its checkout, or its session reads as checked out.
    _ = drainSideQueues(sch);
    var targets: [16]prefix_cache_mod.FlushTarget = undefined;
    const n = diskCaches(sch, &targets, false);
    if (n == 0) return;
    const sw = io_util.Stopwatch.init(sch.io);
    const t = prefix_cache_mod.shutdownPersist(targets[0..n], io_util.MonotonicClock{ .io = sch.io }, deadline_ms);
    const mb = 1024.0 * 1024.0;
    log.info("[disk-cache] shutdown persist: {d} written ({d:.1} MB), {d} left (deadline), {d} left (store declined), {d} skipped checked-out, {d} failures, {d:.1} MB dropped at the deadline, {d:.1} s\n", .{
        t.written,
        @as(f64, @floatFromInt(t.bytes)) / mb,
        t.left_deadline,
        t.left_declined,
        t.checked_out,
        t.failures,
        @as(f64, @floatFromInt(t.dropped_bytes)) / mb,
        @as(f64, @floatFromInt(sw.read())) / std.time.ns_per_s,
    });
}

pub const armShutdownDeadline = @import("../../../host/http/server.zig").armShutdownDeadline;

// ── tests ──

test "staged host bytes include every ready model writer regardless of the current cache view" {
    const disk = @import("ssd/disk_tier.zig");
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const reg = try ModelRegistry.init(testing.allocator, testing.io, null, 4, 0, null);
    defer reg.deinit();
    const a = try reg.registerStub("a", "/a", 1);
    const b = try reg.registerStub("b", "/b", 1);
    for ([_]*LoadedModel{ a, b }) |e| {
        e.state = .ready;
        e.prefix_cache = try testHotCache();
        const hc = e.prefix_cache.?;
        hc.disk = try disk.DiskTier.init(testing.allocator, testing.io, base, e.id, 0, 512);
        hc.disk.?.enableBackgroundWriter();
        const w = hc.disk.?.writer.?;
        w.setPaused(true);
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/staged", .{hc.disk.?.root});
        const bytes = try testing.allocator.dupe(u8, if (e == a) "one" else "second");
        try testing.expectEqual(.accepted, try w.trySubmit(path, bytes));
    }
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = testing.io;
    sch.hot_prefix_cache = b.prefix_cache;
    try testing.expectEqual(@as(u64, 9), stagedHostBytes(&sch));
    a.state = .unloaded;
    try testing.expectEqual(@as(u64, 6), stagedHostBytes(&sch));
    a.state = .ready;
}

test "idle flush reports busy while submitted module files await their writer" {
    const disk = @import("ssd/disk_tier.zig");
    const mlx = @import("../../../mlx/ffi.zig");
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const reg = try ModelRegistry.init(testing.allocator, testing.io, null, 4, 0, null);
    defer reg.deinit();
    const entry = try reg.registerStub("module", "/module", 1);
    entry.state = .ready;
    entry.prefix_cache = try testHotCache();
    const hc = entry.prefix_cache.?;
    const Check = struct {
        fn header(_: ?*anyopaque, _: []const u8, _: *const disk.module_record.Envelope) bool {
            return true;
        }
    };
    hc.disk = try disk.DiskTier.initWithModule(testing.allocator, testing.io, base, "busy", 0, 512, .{ .identity = "fixture", .validate = &Check.header });
    const d = &hc.disk.?;
    d.armTestSpace(1 << 40, 2 << 40);
    d.enableBackgroundWriter();
    const tokens = @as([512]u32, @splat(7));
    const id = (try d.registerModule(.{ .tokens = &tokens, .prefix_key = 1, .has_tools = false, .quant = @import("kv_quant.zig").KVQuantConfig.dense, .header = "header", .tensors = &.{.{ .name = "ring", .dtype = "F32", .shape = &.{1}, .bytes = 4 }} })).?;
    const cpu = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(cpu);
    const arr = mlx.mlx_array_new_data(&[_]f32{1}, &[_]c_int{1}, 1, .float32);
    var tensors = [_]disk.DiskTier.NamedTensor{.{ .key = @constCast("ring"), .arr = arr }};
    // Staging swaps each list entry for its contiguous copy and frees the original.
    defer _ = mlx.mlx_array_free(tensors[0].arr);
    for (0..3) |_| {
        try testing.expectEqual(.progress, try d.stageModulePart(id, &tensors, cpu));
        d.drainWriter();
    }
    d.writer.?.setPaused(true);
    try testing.expectEqual(.done, try d.submitModuleMeta(id));
    hc.disk_dirty = true;
    var xfm: @import("../transformer.zig").Transformer = undefined;
    xfm.s = cpu;
    entry.transformer = &xfm;
    defer entry.transformer = null;
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = testing.io;
    try testing.expectEqual(.busy, flushOwedDiskSlices(&sch));
    try testing.expectEqual(.busy, flushOwedDiskSlices(&sch));
    try testing.expectEqual(disk.module_record.State.submitted, d.moduleState(id).?);
    d.writer.?.setPaused(false);
    d.drainWriter();
    try testing.expectEqual(.done, flushOwedDiskSlices(&sch));
    try testing.expect(d.entryWholeOnDisk(id));
}

test "a parked inference thread keeps the Mac awake only while Neural Engine requests run" {
    var sch: Scheduler = undefined;
    sch.offloaded = .empty;
    defer sch.offloaded.deinit(testing.allocator);
    try testing.expect(!holdsSleepWhileParked(&sch));
    var ane: Slot = undefined;
    try sch.offloaded.append(testing.allocator, &ane);
    try testing.expect(holdsSleepWhileParked(&sch));
}

test "hot-cache view: a model's cache leaving takes the view with it, to the default chat model first" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 4, 0, null);
    defer reg.deinit();
    var chat_cfg = model_mod.ModelConfig{};
    chat_cfg.model_type = "qwen3_5";
    var tts_cfg = model_mod.ModelConfig{};
    tts_cfg.model_type = "kokoro";
    const a = try reg.registerStub("a", "/m/a", 64);
    const b = try reg.registerStub("b", "/m/b", 64);
    const c = try reg.registerStub("c", "/m/c", 64);
    const tts = try reg.registerStub("tts", "/m/tts", 64);
    defer for ([_]*LoadedModel{ a, b, c, tts }) |e| {
        e.config = null;
    };
    for ([_]*LoadedModel{ a, b, c, tts }) |e| {
        e.config = &chat_cfg;
        e.state = .ready;
        e.prefix_cache = try testHotCache();
    }
    tts.config = &tts_cfg;
    reg.default_id = "c";
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = reg.io;

    // Another model leaving does not touch the view.
    sch.hot_prefix_cache = a.prefix_cache.?;
    moveHotCacheViewOff(&sch, b);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, a.prefix_cache.?), sch.hot_prefix_cache);
    // The owner leaving hands it to the default chat model, never to a media model.
    moveHotCacheViewOff(&sch, a);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, c.prefix_cache.?), sch.hot_prefix_cache);
    // With no other chat model resident, the view is dropped, never left on a freed cache.
    b.state = .unloaded;
    a.state = .unloaded;
    moveHotCacheViewOff(&sch, c);
    try testing.expectEqual(@as(?*prefix_cache_mod.HotPrefixCache, null), sch.hot_prefix_cache);
}

/// A heap cache the registry's teardown frees, as a load allocates one.
fn testHotCache() !*prefix_cache_mod.HotPrefixCache {
    const hc = try testing.allocator.create(prefix_cache_mod.HotPrefixCache);
    hc.* = prefix_cache_mod.HotPrefixCache.init(testing.allocator, 4);
    return hc;
}

fn budget200Mb(_: *model_mod.ModelConfig, _: u64, _: BudgetRevise, idle: *u64) u64 {
    idle.* = 0;
    return 200 << 20;
}

test "hot-cache view: a revise that follows an unload raises a budget but never lowers it" {
    var reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 4, 0, null);
    defer reg.deinit();
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "qwen3_5";
    const m = try reg.registerStub("m", "/m/m", 64);
    defer m.config = null;
    m.config = &cfg;
    m.state = .ready;
    m.prefix_cache = try testHotCache();
    const hc = m.prefix_cache.?;
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = reg.io;
    sch.allocator = testing.allocator;
    sch.current_model = null;
    sch.hot_prefix_cache = null;
    sch.prefix_cache_mem_resolver = budget200Mb;
    sch.prefix_cache_mem_bytes = 0;
    sch.metrics = null;
    sch.digest_mu = .init;
    sch.hot_cache_digests = &.{};
    sch.digest_residency = 0;
    sch.resident_hot_cache_bytes = std.atomic.Value(u64).init(0);
    sch.reclaimable_hot_cache_bytes = std.atomic.Value(u64).init(0);
    sch.resident_live_kv_bytes = .init(0);
    sch.queue_mu = .init;
    sch.decoding = .empty;

    hc.max_kv_bytes = 600 << 20;
    reviseHotCacheBudgets(&sch, true);
    try testing.expectEqual(@as(u64, 600 << 20), hc.max_kv_bytes);
    hc.max_kv_bytes = 100 << 20;
    reviseHotCacheBudgets(&sch, true);
    try testing.expectEqual(@as(u64, 200 << 20), hc.max_kv_bytes);
    // A load's revise is honest both ways.
    hc.max_kv_bytes = 600 << 20;
    reviseHotCacheBudgets(&sch, false);
    try testing.expectEqual(@as(u64, 200 << 20), hc.max_kv_bytes);
}

test "module live residency is counted once per model and cannot be credited as evictable cache" {
    const abi = @import("../../../abi/plugin.zig");
    const Fake = struct {
        fn bytes(_: *anyopaque) callconv(.c) u64 {
            return 64;
        }
        const table: abi.ModelTable = .{
            .name = .of("live_residency_fixture"),
            .claims = undefined,
            .parse = undefined,
            .free_parsed = undefined,
            .bill = undefined,
            .load = undefined,
            .unload = undefined,
            .forward = undefined,
            .cache = &.{ .state_bytes = &bytes },
            .weights = &.{},
        };
    };
    const reg = try ModelRegistry.init(testing.allocator, std.Io.Threaded.global_single_threaded.io(), null, 4, 0, null);
    defer reg.deinit();
    var token: u8 = 0;
    var bound: plugin_driver.Bound = .{ .table = &Fake.table, .model = &token };
    var xfm: @import("../transformer.zig").Transformer = undefined;
    xfm.plugin = &bound;
    const m = try reg.registerStub("m", "/m", 1);
    m.state = .ready;
    m.transformer = &xfm;
    defer m.transformer = null;
    var sch: Scheduler = undefined;
    sch.registry = reg;
    sch.io = reg.io;
    sch.allocator = testing.allocator;
    sch.metrics = null;
    sch.hot_prefix_cache = null;
    sch.hot_cache_digests = &.{};
    sch.digest_mu = .init;
    sch.queue_mu = .init;
    sch.decoding = .empty;
    sch.resident_hot_cache_bytes = .init(0);
    sch.reclaimable_hot_cache_bytes = .init(0);
    sch.resident_live_kv_bytes = .init(0);
    publishHotCacheResidency(&sch);
    try testing.expectEqual(@as(u64, 0), sch.resident_hot_cache_bytes.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), sch.reclaimable_hot_cache_bytes.load(.monotonic));
    try testing.expectEqual(@as(u64, 0), sch.resident_live_kv_bytes.load(.monotonic));
    for (0..2) |_| {
        const module_bytes = moduleStateBytes(&sch);
        sch.queue_mu.lockUncancelable(sch.io);
        @import("../step.zig").publishLiveKvResidency(&sch, null, module_bytes);
        sch.queue_mu.unlock(sch.io);
        try testing.expectEqual(@as(u64, 64), sch.resident_live_kv_bytes.load(.monotonic));
    }
    try testing.expectEqual(@as(u64, 0), sch.resident_hot_cache_bytes.load(.monotonic));
}
