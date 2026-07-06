const std = @import("std");
const builtin = @import("builtin");
const is_macos = builtin.os.tag == .macos;

// ── macOS Mach externs ──

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

// ── IOKit / CoreFoundation externs ──

extern "c" fn IOServiceMatching(name: [*:0]const u8) ?*anyopaque;
extern "c" fn IOServiceGetMatchingServices(port: u32, matching: ?*anyopaque, iter: *u32) i32;
extern "c" fn IOIteratorNext(iter: u32) u32;
extern "c" fn IORegistryEntryCreateCFProperties(entry: u32, props: *?*anyopaque, alloc: ?*anyopaque, opts: u32) i32;
extern "c" fn IOObjectRelease(obj: u32) i32;
extern "c" fn CFDictionaryGetValue(dict: ?*const anyopaque, key: ?*const anyopaque) ?*const anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: ?*anyopaque, s: [*:0]const u8, enc: u32) ?*const anyopaque;
extern "c" fn CFNumberGetValue(num: ?*const anyopaque, typ: u32, out: *anyopaque) u8;
extern "c" fn CFRelease(cf: ?*const anyopaque) void;

// ── Mach struct layouts (extern = C ABI) ──

const TaskBasicInfo = extern struct {
    virtual_size: u64,
    resident_size: u64,
    resident_size_max: u64,
    user_time_sec: i32,
    user_time_usec: i32,
    sys_time_sec: i32,
    sys_time_usec: i32,
    policy: i32,
    suspend_count: i32,
};

const CpuLoadInfo = extern struct {
    ticks: [4]u32, // user, system, idle, nice
};

const VmStats64 = extern struct {
    free_count: u32,
    active_count: u32,
    inactive_count: u32,
    wire_count: u32,
    zero_fill_count: u64,
    reactivations: u64,
    pageins: u64,
    pageouts: u64,
    faults: u64,
    cow_faults: u64,
    lookups: u64,
    hits: u64,
    purges: u64,
    purgeable_count: u32,
    speculative_count: u32,
    decompressions: u64,
    compressions: u64,
    swapins: u64,
    swapouts: u64,
    compressor_page_count: u32,
    throttled_count: u32,
    external_page_count: u32,
    internal_page_count: u32,
    total_uncompressed_pages_in_compressor: u64,
};

// ── CPU delta tracking (module-level state) ──
var prev_ticks: [4]u64 = .{0} ** 4;

// ── Public metric helpers ──

pub fn getAppRssMb() u32 {
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

/// Bytes of physical memory available for new allocation without heavy
/// Pure: bytes available for a new large allocation given the live page counts.
///
/// Subtracts the genuinely non-reclaimable set: `wired` (pinned), `compressor`
/// (already-compressed app data), and `internal` (anonymous app pages — crucially
/// INCLUDING a resident MLX model). File-backed cache (`external`) plus
/// free/speculative/purgeable pages are NOT subtracted: macOS evicts them the
/// instant a big allocation lands, so they don't block a load. That keeps a 12B
/// (~7.7 GB) loading on a 16 GB Mac that shows only ~7.8 GB *instantaneous* free
/// (the rest is reclaimable file cache). It also fixes the #45 OOM: a prior model
/// still resident lives in the anonymous (`internal`) set, NOT necessarily in
/// `wired` (verified live: a 5 GB resident model with only ~2.8 GB total wired) —
/// so counting `internal` makes a second large load correctly fail the guard.
/// Slightly conservative (wired-anonymous pages can appear in both `wired` and
/// `internal`), which is the safe direction for an OOM guard
/// (`--skip-mem-preflight` overrides). Returns 0 when total is 0 or
/// used ≥ total (a failed query must never block).
fn computeAvailableBytes(total_mem: u64, wire_pages: u64, compressor_pages: u64, internal_pages: u64, page: u64) u64 {
    const used: u64 = (wire_pages + compressor_pages + internal_pages) * page;
    if (total_mem == 0 or used >= total_mem) return 0;
    return total_mem - used;
}

extern "c" fn os_proc_available_memory() usize;

/// Per-process memory headroom before jetsam (iOS only — the entitlement-
/// aware figure that actually governs whether a big allocation survives).
/// Returns 0 on macOS, where the concept doesn't apply; the symbol is only
/// referenced on iOS builds so macOS links are unaffected.
pub fn getProcAvailableMemBytes() u64 {
    if (comptime builtin.os.tag != .ios) return 0;
    return @intCast(os_proc_available_memory());
}

/// Total physical RAM (hw.memsize; works on macOS and iOS). 0 on failure.
pub fn getTotalMemBytes() u64 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

pub fn getAvailableMemBytes() u64 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    return computeAvailableBytes(total_mem, vm.wire_count, vm.compressor_page_count, vm.internal_page_count, page);
}

test "computeAvailableBytes counts the resident anon set, not file cache" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page; // pages per GB

    // 16 GB Mac, light anon load: 3 GB wired, 1 GB compressed, 2 GB anonymous app
    // pages; the remaining ~10 GB is free + reclaimable file cache, which must NOT
    // count against availability. Available = 16 − (3+1+2) = 10 GB. (The old
    // `active`-subtracting formula counted file cache and wrongly refused loads
    // that fit; later dropping `active` entirely wrongly ignored resident models.)
    try std.testing.expectEqual(@as(u64, 10 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 2 * ppg, page));

    // #45 OOM guard: a prior 7 GB model is resident. It lives in the anonymous
    // (`internal`) set — here 9 GB = 2 GB apps + 7 GB model — NOT in `wired`. So
    // available = 16 − (3+1+9) = 3 GB, and a second 7 GB load is correctly refused.
    try std.testing.expectEqual(@as(u64, 3 * GB), computeAvailableBytes(16 * GB, 3 * ppg, 1 * ppg, 9 * ppg, page));

    // Degenerate guards: failed query (total 0) and used ≥ total → 0, never block.
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(0, 1, 1, 1, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(8 * GB, 4 * ppg, 0, 5 * ppg, page));
}

pub fn getSysMemPct() u32 {
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return 0;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return 0;

    const used: u64 = (@as(u64, vm.active_count) + vm.wire_count + vm.compressor_page_count) * page;
    if (total_mem == 0) return 0;
    return @intCast(used * 100 / total_mem);
}

pub fn getCpuPct() u32 {
    var info = std.mem.zeroes(CpuLoadInfo);
    var count: u32 = 4;
    if (host_statistics(mach_host_self(), 3, @ptrCast(&info), &count) != 0) return 0;

    var total: u64 = 0;
    var idle: u64 = 0;
    for (0..4) |i| {
        const cur: u64 = info.ticks[i];
        const delta = cur -| prev_ticks[i];
        total += delta;
        if (i == 2) idle = delta;
        prev_ticks[i] = cur;
    }
    if (total == 0) return 0;
    return @intCast((total - idle) * 100 / total);
}

pub fn getGpuPct() u32 {
    // IOKit's IOServiceMatching/AGXAccelerator path is macOS-only; the symbols
    // aren't in the public iOS SDK (and apps are sandboxed from the GPU service
    // registry anyway). On iOS we report 0 — the value is only a log-line stat.
    if (comptime !is_macos) return 0;
    const matching = IOServiceMatching("AGXAccelerator") orelse return 0;
    var iter: u32 = 0;
    if (IOServiceGetMatchingServices(0, matching, &iter) != 0) return 0;
    defer _ = IOObjectRelease(iter);

    const entry = IOIteratorNext(iter);
    if (entry == 0) return 0;
    defer _ = IOObjectRelease(entry);

    var props: ?*anyopaque = null;
    if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) return 0;
    defer if (props) |p| CFRelease(p);

    const perf = cfDictGet(props, "PerformanceStatistics") orelse return 0;
    const util = cfDictGet(perf, "Device Utilization %") orelse return 0;

    var value: i64 = 0;
    _ = CFNumberGetValue(util, 4, @ptrCast(&value));
    return if (value >= 0 and value <= 100) @intCast(value) else 0;
}

fn cfDictGet(dict: ?*const anyopaque, key_name: [*:0]const u8) ?*const anyopaque {
    const key = CFStringCreateWithCString(null, key_name, 0x08000100) orelse return null;
    defer CFRelease(key);
    return CFDictionaryGetValue(dict, key);
}
