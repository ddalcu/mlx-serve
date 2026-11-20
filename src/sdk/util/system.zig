//! What the machine has and how loaded it is: process and system memory, the OS reserve, memory pressure, CPU and GPU
//! load, GPU cores (Mach, IOKit, /proc, Win32).

const std = @import("std");
const crt = @import("crt.zig");
const os_mutex = @import("os_mutex.zig");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
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

/// task_vm_info truncated through phys_footprint (rev1). Field order matches
/// <mach/task_info.h> exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 38 ==
/// TASK_VM_INFO_REV1_COUNT, so the kernel fills through phys_footprint without
/// overrunning the buffer.
const TaskVmInfo = extern struct {
    virtual_size: u64,
    region_count: i32,
    page_size: i32,
    resident_size: u64,
    resident_size_peak: u64,
    device: u64,
    device_peak: u64,
    internal: u64,
    internal_peak: u64,
    external: u64,
    external_peak: u64,
    reusable: u64,
    reusable_peak: u64,
    purgeable_volatile_pmap: u64,
    purgeable_volatile_resident: u64,
    purgeable_volatile_virtual: u64,
    compressed: u64,
    compressed_peak: u64,
    compressed_lifetime: u64,
    phys_footprint: u64,
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
var prev_ticks: [4]u64 = @splat(0);

// ── Public metric helpers ──

pub fn getAppRssMb() u32 {
    if (comptime !builtin.os.tag.isDarwin())
        return @intCast(linuxProcStatusKib("VmRSS:") / 1024);
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

/// Process physical memory footprint in MB (TASK_VM_INFO flavor 22). Unlike
/// resident_size, this includes MLX's Metal/IOKit + compressed memory — the
/// only figure that reflects a loaded model's true footprint on Apple Silicon.
/// On Linux the honest analog is VmRSS (unified-memory footprint has no
/// equivalent); swap-paged-out pages are invisible to it.
pub fn getAppMemFootprintMb() u32 {
    if (comptime !builtin.os.tag.isDarwin())
        return @intCast(linuxProcStatusKib("VmRSS:") / 1024);
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32); // 38 = TASK_VM_INFO_REV1_COUNT
    if (task_info(mach_task_self_, 22, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.phys_footprint / (1024 * 1024));
}

/// Bytes the OS will grant a new wiring: free (less speculative — free_count
/// already contains them and `external` carries them back) plus file-backed plus
/// purgeable (anonymous caches macOS drops before it compresses anything), capped
/// by the RAM outside wired + compressor. 0 on a failed query.
pub fn computeAvailableBytes(total_mem: u64, free_pages: u64, speculative_pages: u64, external_pages: u64, purgeable_pages: u64, wire_pages: u64, compressor_pages: u64, page: u64) u64 {
    if (total_mem == 0) return 0;
    const reclaimable: u64 = (free_pages -| speculative_pages) +| external_pages +| purgeable_pages;
    const outside: u64 = total_mem -| (wire_pages +| compressor_pages) *| page;
    return @min(reclaimable *| page, outside);
}

/// What a model LOAD can count on: the RAM outside wired and compressor memory, less the OS reserve. macOS
/// compresses or swaps any app's pages to make room for a wiring, so only wired memory (another loaded
/// model, the kernel) and the compressor's own pages bind. Loads and request admission read it on macOS
/// (`getLoadRoomBytes`); the hot cache and the app's meter keep `computeAvailableBytes`. 0 on a failed query.
pub fn computeLoadAvailableBytes(total_mem: u64, wire_pages: u64, compressor_pages: u64, page: u64, reserve: u64) u64 {
    if (total_mem == 0) return 0;
    return (total_mem -| (wire_pages +| compressor_pages) *| page) -| reserve;
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

/// Where weights live in device memory (`mlx.weightsInDeviceMemory`: CUDA on Windows, WSL) the
/// GPU's memory is what every gate bills, as unified memory is on a Mac; elsewhere the host's.
pub fn memoryPool(device_weights: bool, device: u64, host: u64) u64 {
    return if (device_weights) device else host;
}

pub fn getTotalMemBytes() u64 {
    return memoryPool(mlx.weightsInDeviceMemory(), mlx.maxRecommendedWorkingSet(), hostTotalMemBytes());
}

/// The box's memory in use outside the file cache: wired + anonymous - purgeable + compressor (macOS's counts; the
/// `memory.tsv` `used_gb` figure). Unlike `total - available`, it does not move with how macOS counts recently read
/// file pages. Elsewhere `total - available`.
pub fn getNonFileMemBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return hostTotalMemBytes() -| hostAvailableMemBytes();
    const v = darwinVm() orelse return 0;
    return nonFileBytes(v.vm, v.page);
}

/// A planted `reading_override` from byte counts (tests): the rest of the box free.
pub fn plantVm(total: u64, page: u64, b: struct { wired: u64, anonymous: u64, compressor: u64, purgeable: u64, file: u64 }) @TypeOf(reading_override.?) {
    var vm = std.mem.zeroes(VmStats64);
    vm.wire_count = @intCast(b.wired / page);
    vm.internal_page_count = @intCast(b.anonymous / page);
    vm.compressor_page_count = @intCast(b.compressor / page);
    vm.purgeable_count = @intCast(b.purgeable / page);
    vm.external_page_count = @intCast(b.file / page);
    vm.free_count = @intCast((total -| b.wired -| b.anonymous -| b.compressor -| b.file) / page);
    return .{ .vm = .{ .total = total, .page = page, .vm = vm }, .footprint = 0, .now_ms = 1000 };
}

fn nonFileBytes(vm: VmStats64, page: u64) u64 {
    const pages = (@as(u64, vm.wire_count) +| vm.internal_page_count +| vm.compressor_page_count) -| vm.purgeable_count;
    return pages *| page;
}

pub fn getAvailableMemBytes() u64 {
    const host = hostAvailableMemBytes();
    return memoryPool(mlx.weightsInDeviceMemory(), mlx.freeDeviceMemory(), host +| returnedNotShown(host));
}

/// macOS shows memory MLX handed back about a second late, while a new allocation can take it at once: a reading
/// taken right after an eviction or an unload refused work that fit. This carries what MLX returned between two
/// close readings until the OS figure shows it, or MLX takes it again.
pub const ReturnLag = struct {
    /// The OS figure catches up well inside this; a credit older than it is dropped.
    pub const WINDOW_MS: i64 = 2000;

    at_ms: i64 = 0,
    /// MLX's active memory and buffer pool at the last reading: it shrinks only by what went back to the OS.
    footprint: u64 = 0,
    os_free: u64 = 0,
    pending: u64 = 0,
    pending_at_ms: i64 = 0,

    /// The bytes to add to `os_free` at this reading.
    pub fn credit(self: *ReturnLag, now_ms: i64, footprint: u64, os_free: u64) u64 {
        if (now_ms - self.pending_at_ms > WINDOW_MS) self.pending = 0;
        if (self.at_ms != 0 and now_ms - self.at_ms <= WINDOW_MS and footprint < self.footprint) {
            self.pending +|= self.footprint - footprint;
            self.pending_at_ms = now_ms;
        }
        self.pending -|= os_free -| self.os_free;
        self.pending -|= footprint -| self.footprint;
        self.at_ms = now_ms;
        self.footprint = footprint;
        self.os_free = os_free;
        return self.pending;
    }
};

const ReturnLagState = struct { mtx: os_mutex.Mutex = .{}, lag: ReturnLag = .{} };
pub const return_lag_shared = @import("shared.zig").Shared(ReturnLagState, "mem_return_lag", .{});

fn returnedNotShown(os_free: u64) u64 {
    if (comptime !is_macos) return 0;
    if (os_free == 0) return 0;
    var active: usize = 0;
    var pool: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&pool);
    var footprint: u64 = @as(u64, active) +| pool;
    var now_ms = crt.monotonicMs();
    if (reading_override) |r| {
        footprint = r.footprint;
        now_ms = r.now_ms;
    }
    const st = return_lag_shared.get();
    st.mtx.lock();
    defer st.mtx.unlock();
    return st.lag.credit(now_ms, footprint, os_free);
}

/// Test seam: the kernel's counts, MLX's footprint and the clock, read in place of the machine's.
pub var reading_override: ?struct { vm: DarwinVm, footprint: u64, now_ms: i64 } = null;

/// `getAvailableMemBytes` for a model load (`computeLoadAvailableBytes`). Other hosts' readings
/// already count what their OS reclaims (Linux `MemAvailable`, Windows `ullAvailPhys`).
pub fn getLoadAvailableMemBytes() u64 {
    return memoryPool(mlx.weightsInDeviceMemory(), mlx.freeDeviceMemory(), hostLoadAvailableMemBytes());
}

/// RAM a new wiring or allocation can take beside everything resident: `getLoadAvailableMemBytes` less the
/// OS reserve, which macOS's figure already leaves out.
pub fn getLoadRoomBytes() u64 {
    const avail = getLoadAvailableMemBytes();
    if (comptime builtin.os.tag == .macos) return avail;
    return avail -| osReserveBytes(getTotalMemBytes(), mlx.weightsInDeviceMemory());
}

/// The kernel's memory-pressure level: 1 normal, 2 warning, 4 critical; 0 where the OS has no such reading.
pub fn memoryPressureLevel() u32 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var level: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", @ptrCast(&level), &len, null, 0) != 0) return 0;
    return if (level > 0) @intCast(level) else 0;
}

/// Warning or worse: the OS is already compressing and swapping to make room.
pub fn underMemoryPressure(level: u32) bool {
    return level >= 2;
}

/// Total physical RAM (hw.memsize on Darwin; /proc/meminfo on Linux). 0 on failure.
fn hostTotalMemBytes() u64 {
    if (comptime builtin.os.tag == .windows) return windowsMemStatus().ullTotalPhys;
    if (comptime !builtin.os.tag.isDarwin()) return linuxMemInfoKib("MemTotal:") * 1024;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

/// Free RAM the plan never touches. MLX wires what it allocates, so planning down to the last
/// free page leaves the OS nothing to reclaim: a 16 GB Mac died on wired memory (15.4 GB wired,
/// 14 MB free) with every request admitted.
/// `device_pool`: the memory is a GPU's own (`mlx.weightsInDeviceMemory`), whose free reading
/// already counts every process, so no OS share is held back.
pub fn osReserveBytes(total_ram: u64, device_pool: bool) u64 {
    if (os_reserve_override) |v| return v;
    if (device_pool) return 0;
    return std.math.clamp(total_ram / 8, 2 << 30, 8 << 30);
}

/// `--os-reserve-gib N` in bytes; 0 turns the reserve off. Null = the automatic eighth.
pub var os_reserve_override: ?u64 = null;

pub fn parseOsReserveGib(raw: []const u8) error{InvalidOsReserve}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidOsReserve;
    if (n > 64) return error.InvalidOsReserve;
    return @as(u64, n) << 30;
}

fn hostAvailableMemBytes() u64 {
    if (comptime builtin.os.tag == .windows) return windowsMemStatus().ullAvailPhys;
    if (comptime !builtin.os.tag.isDarwin()) return linuxMemInfoKib("MemAvailable:") * 1024;
    return darwinFree(darwinVm() orelse return 0);
}

fn darwinFree(v: DarwinVm) u64 {
    return computeAvailableBytes(v.total, v.vm.free_count, v.vm.speculative_count, v.vm.external_page_count, v.vm.purgeable_count, v.vm.wire_count, v.vm.compressor_page_count, v.page);
}

fn hostLoadAvailableMemBytes() u64 {
    if (comptime builtin.os.tag != .macos) return hostAvailableMemBytes();
    const v = darwinVm() orelse return 0;
    const room = computeLoadAvailableBytes(v.total, v.vm.wire_count, v.vm.compressor_page_count, v.page, osReserveBytes(v.total, false));
    // MLX's buffers are wired, so this figure shows a return as late as the free one. The credit follows the free
    // figure from every reader: fed this one here and the free one elsewhere, each reading would wipe the other's.
    return room +| returnedNotShown(darwinFree(v));
}

const DarwinVm = struct { total: u64, page: u64, vm: VmStats64 };

/// Total RAM, page size and the kernel's page counts; null on a failed query.
fn darwinVm() ?DarwinVm {
    if (reading_override) |r| return r.vm;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return null;

    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return null;

    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return null;
    return .{ .total = total_mem, .page = page, .vm = vm };
}

const MemoryStatusEx = extern struct {
    dwLength: u32 = @sizeOf(MemoryStatusEx),
    dwMemoryLoad: u32 = 0,
    ullTotalPhys: u64 = 0,
    ullAvailPhys: u64 = 0,
    ullTotalPageFile: u64 = 0,
    ullAvailPageFile: u64 = 0,
    ullTotalVirtual: u64 = 0,
    ullAvailVirtual: u64 = 0,
    ullAvailExtendedVirtual: u64 = 0,
};
extern "kernel32" fn GlobalMemoryStatusEx(status: *MemoryStatusEx) callconv(.winapi) i32;

/// Zeroed (callers' "unknown") when the call fails.
fn windowsMemStatus() MemoryStatusEx {
    var s: MemoryStatusEx = .{};
    if (GlobalMemoryStatusEx(&s) == 0) return .{};
    return s;
}

/// One `/proc/meminfo` field in KiB ("MemTotal:", "MemAvailable:", …).
/// 0 when missing or unreadable — callers treat 0 as "unknown".
fn linuxMemInfoKib(key: []const u8) u64 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile("/proc/meminfo", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        const val = std.mem.trim(u8, line[key.len..], " \t");
        const end = std.mem.indexOf(u8, val, " kB") orelse val.len;
        return std.fmt.parseInt(u64, std.mem.trim(u8, val[0..end], " \t"), 10) catch 0;
    }
    return 0;
}

/// One `/proc/self/status` field in KiB ("VmRSS:", "VmHWM:", …). 0 unknown.
fn linuxProcStatusKib(key: []const u8) u64 {
    var buf: [8192]u8 = undefined;
    const n = readProcFile("/proc/self/status", &buf) orelse return 0;
    var it = std.mem.splitScalar(u8, buf[0..n], '\n');
    while (it.next()) |line| {
        if (!std.mem.startsWith(u8, line, key)) continue;
        const val = std.mem.trim(u8, line[key.len..], " \t");
        const end = std.mem.indexOf(u8, val, " kB") orelse val.len;
        return std.fmt.parseInt(u64, std.mem.trim(u8, val[0..end], " \t"), 10) catch 0;
    }
    return 0;
}

/// Whole-file read of a small procfs file. Returns null when missing/unreadable.
fn readProcFile(path: []const u8, buf: []u8) ?usize {
    if (path.len >= std.fs.max_path_bytes) return null;
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    @memcpy(pbuf[0..path.len], path);
    pbuf[path.len] = 0;
    const fd = crt.openRead(pbuf[0..path.len :0].ptr);
    if (fd < 0) return null;
    defer crt.close(fd);
    const n = crt.read(fd, buf);
    if (n <= 0) return null;
    return @intCast(n);
}

test "ReturnLag: bytes MLX returned between two close readings are credited until the OS figure shows them" {
    const GB: u64 = 1 << 30;
    var lag: ReturnLag = .{};
    // The first reading has no baseline.
    try std.testing.expectEqual(@as(u64, 0), lag.credit(1000, 30 * GB, 10 * GB));
    // An eviction returned 4 GB and the OS figure has not moved: all 4 are credited.
    try std.testing.expectEqual(4 * GB, lag.credit(1001, 26 * GB, 10 * GB));
    // The OS shows 1 GB of it 200 ms later, then the rest.
    try std.testing.expectEqual(3 * GB, lag.credit(1200, 26 * GB, 11 * GB));
    try std.testing.expectEqual(@as(u64, 0), lag.credit(2200, 26 * GB, 14 * GB));
    // A return the OS shows within the same pair of readings needs no credit.
    try std.testing.expectEqual(@as(u64, 0), lag.credit(2300, 24 * GB, 16 * GB));
    // What MLX takes again is no longer room.
    try std.testing.expectEqual(4 * GB, lag.credit(2400, 20 * GB, 16 * GB));
    try std.testing.expectEqual(1 * GB, lag.credit(2401, 23 * GB, 16 * GB));
    // A figure that never rises stops being credited after the window.
    try std.testing.expectEqual(@as(u64, 0), lag.credit(2401 + ReturnLag.WINDOW_MS + 1, 23 * GB, 16 * GB));
    // A drop seen across readings further apart than the window had time to show: no credit.
    try std.testing.expectEqual(@as(u64, 0), lag.credit(20_000, 19 * GB, 16 * GB));
}

test "ReturnLag sees a buffer MLX frees past its pool as returned" {
    // Bar: with the pool full a freed buffer leaves MLX's footprint, which is what the credit reads. The OS figure is
    // held fixed here: other processes move the real one.
    if (!is_macos or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, 0); // a free goes back to the OS, as with the pool at its cap
    defer {
        var t: usize = 0;
        _ = mlx.mlx_set_cache_limit(&t, prev);
    }
    const footprint = struct {
        fn now() u64 {
            var active: usize = 0;
            var pool: usize = 0;
            _ = mlx.mlx_get_active_memory(&active);
            _ = mlx.mlx_get_cache_memory(&pool);
            return @as(u64, active) +| pool;
        }
    }.now;
    const bytes: u64 = 1 << 30;
    var a = mlx.mlx_array_new();
    const shape = [_]c_int{1 << 28};
    const s = mlx.gpuStream();
    try mlx.check(mlx.mlx_ones(&a, &shape, 1, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a));
    // The eval's command buffer holds the buffer until its completion ran.
    _ = mlx.mlx_synchronize(s);
    var lag: ReturnLag = .{};
    _ = lag.credit(1000, footprint(), 8 << 30);
    _ = mlx.mlx_array_free(a);
    try std.testing.expect(lag.credit(1001, footprint(), 8 << 30) >= bytes - bytes / 8);
}

test "the load figure carries what MLX just returned, and a free-figure reading between does not wipe it" {
    if (!is_macos) return error.SkipZigTest;
    const GB: u64 = 1 << 30;
    const page: u64 = 16384;
    const st = return_lag_shared.get();
    st.lag = .{};
    defer {
        st.lag = .{};
        reading_override = null;
    }
    var vm = std.mem.zeroes(VmStats64);
    vm.free_count = @intCast(20 * GB / page);
    vm.wire_count = @intCast(80 * GB / page);
    reading_override = .{ .vm = .{ .total = 128 * GB, .page = page, .vm = vm }, .footprint = 70 * GB, .now_ms = 1000 };
    const load_before = getLoadAvailableMemBytes();
    const free_before = getAvailableMemBytes();
    // An eviction returned 4 GB; the kernel's counts have not moved yet.
    reading_override.?.footprint = 66 * GB;
    reading_override.?.now_ms = 1001;
    try std.testing.expectEqual(load_before + 4 * GB, getLoadAvailableMemBytes());
    reading_override.?.now_ms = 1002;
    try std.testing.expectEqual(free_before + 4 * GB, getAvailableMemBytes());
    reading_override.?.now_ms = 1003;
    try std.testing.expectEqual(load_before + 4 * GB, getLoadAvailableMemBytes());
}

test "getNonFileMemBytes: the file pages macOS keeps do not move it; wired, anonymous and compressor do, purgeable not" {
    if (!is_macos) return error.SkipZigTest;
    const GB: u64 = 1 << 30;
    const page: u64 = 16384;
    defer reading_override = null;
    // one box state, twice: the second with 7 GB more file-backed pages, taken from free
    var low = std.mem.zeroes(VmStats64);
    low.wire_count = @intCast(20 * GB / page);
    low.internal_page_count = @intCast(10 * GB / page);
    low.compressor_page_count = @intCast(1 * GB / page);
    low.purgeable_count = @intCast(GB / 2 / page);
    low.external_page_count = @intCast(4 * GB / page);
    low.free_count = @intCast(80 * GB / page);
    var high = low;
    high.external_page_count = @intCast(11 * GB / page);
    high.free_count = @intCast(73 * GB / page);
    reading_override = .{ .vm = .{ .total = 128 * GB, .page = page, .vm = low }, .footprint = 0, .now_ms = 1000 };
    const at_low = getNonFileMemBytes();
    reading_override = .{ .vm = .{ .total = 128 * GB, .page = page, .vm = high }, .footprint = 0, .now_ms = 1000 };
    try std.testing.expectEqual(at_low, getNonFileMemBytes());
    try std.testing.expectEqual(20 * GB + 10 * GB + GB - GB / 2, at_low);
    var more = high;
    more.internal_page_count += @intCast(GB / page);
    try std.testing.expectEqual(at_low + GB, nonFileBytes(more, page));
}

test "memoryPool: device-memory weights bill VRAM, every other platform its host reading" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(15 * GB, memoryPool(true, 15 * GB, 9 * GB)); // Windows: 13 GB fits the card
    try std.testing.expectEqual(20 * GB, memoryPool(false, 15 * GB, 20 * GB)); // managed memory pages to RAM
}

test "osReserveBytes: an eighth of RAM, never under 2 GB or over 8 GB; none of a GPU's own memory" {
    const gb: u64 = 1 << 30;
    try std.testing.expectEqual(2 * gb, osReserveBytes(8 * gb, false));
    try std.testing.expectEqual(2 * gb, osReserveBytes(16 * gb, false));
    try std.testing.expectEqual(4 * gb, osReserveBytes(32 * gb, false));
    try std.testing.expectEqual(8 * gb, osReserveBytes(128 * gb, false));
    try std.testing.expectEqual(@as(u64, 0), osReserveBytes(16 * gb, true)); // free VRAM already counts every process
    os_reserve_override = try parseOsReserveGib("0");
    defer os_reserve_override = null;
    try std.testing.expectEqual(@as(u64, 0), osReserveBytes(16 * gb, false));
    try std.testing.expectError(error.InvalidOsReserve, parseOsReserveGib("lots"));
}

test "computeAvailableBytes sums the reclaimable classes, capped by wired headroom" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page; // pages per GB

    // free_count already contains speculative, and speculative pages are
    // file-backed inside external — the sum carries them once.
    try std.testing.expectEqual(@as(u64, 32 * GB), computeAvailableBytes(128 * GB, 8 * ppg, 2 * ppg, 26 * ppg, 0, 83 * ppg, ppg, page));

    // The cap: wired + compressor bound the figure, however the wired pages
    // are counted elsewhere.
    try std.testing.expectEqual(@as(u64, 6 * GB), computeAvailableBytes(16 * GB, 6 * ppg, 0, 6 * ppg, 0, 9 * ppg, ppg, page));

    // #45 OOM guard, in new units: a resident 7 GB model lives in the anonymous
    // set, NOT in wired — it consumed the free/external pages (16 GB box: 9 GB
    // anon + 3 wired + 1 comp), so only 3 GB of reclaimable classes remain.
    try std.testing.expectEqual(@as(u64, 3 * GB), computeAvailableBytes(16 * GB, 2 * ppg, 0, 1 * ppg, 0, 3 * ppg, 1 * ppg, page));

    // Saturating: disagreeing counters never underflow; a failed query (total 0)
    // or a box with no headroom (wired + compressor ≥ total) reports 0.
    try std.testing.expectEqual(@as(u64, 4 * GB), computeAvailableBytes(16 * GB, ppg, 5 * ppg, 4 * ppg, 0, 0, 0, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(0, 1, 1, 1, 0, 1, 1, page));
    try std.testing.expectEqual(@as(u64, 0), computeAvailableBytes(16 * GB, 6 * ppg, 0, 6 * ppg, 0, 16 * ppg, 0, page));

    // Purgeable pages are anonymous caches the OS discards before compressing
    // anything: a laptop's browsers park gigabytes there (#932).
    try std.testing.expectEqual(@as(u64, 7 * GB), computeAvailableBytes(16 * GB, 2 * ppg, 0, 1 * ppg, 4 * ppg, 3 * ppg, 1 * ppg, page));
}

test "computeLoadAvailableBytes is the RAM outside wired and compressor memory, less the OS reserve" {
    const GB: u64 = 1024 * 1024 * 1024;
    const page: u64 = 16384;
    const ppg: u64 = GB / page;

    // 128 GB box, 8 wired + 2 compressor, 8 reserved: 110 GB, whatever the apps hold, since macOS moves them for a wiring.
    try std.testing.expectEqual(@as(u64, 110 * GB), computeLoadAvailableBytes(128 * GB, 8 * ppg, 2 * ppg, page, 8 * GB));
    // Another model wired at 70 GB leaves 40: wired memory is the one thing nothing moves.
    try std.testing.expectEqual(@as(u64, 40 * GB), computeLoadAvailableBytes(128 * GB, 78 * ppg, 2 * ppg, page, 8 * GB));
    // Past the RAM: 0, never a wrapped figure.
    try std.testing.expectEqual(@as(u64, 0), computeLoadAvailableBytes(16 * GB, 13 * ppg, 2 * ppg, page, 2 * GB));
    // A failed query reads 0.
    try std.testing.expectEqual(@as(u64, 0), computeLoadAvailableBytes(0, 0, 0, page, 8 * GB));
}

pub fn getSysMemPct() u32 {
    if (comptime !builtin.os.tag.isDarwin()) {
        // (total - available) / total, from the kernel's MemAvailable.
        const total = linuxMemInfoKib("MemTotal:");
        if (total == 0) return 0;
        const avail = linuxMemInfoKib("MemAvailable:");
        return @intCast((total -| avail) * 100 / total);
    }
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
    if (comptime !builtin.os.tag.isDarwin()) {
        // /proc/stat aggregate line: "cpu  user nice system idle iowait irq
        // softirq steal ...". Delta over the calls, idle = nice-adjusted idle
        // columns, mirroring the Mach ticks loop below.
        var buf: [512]u8 = undefined;
        const n = readProcFile("/proc/stat", &buf) orelse return 0;
        const line_end = std.mem.indexOfScalar(u8, buf[0..n], '\n') orelse n;
        var it = std.mem.tokenizeAny(u8, buf[0..line_end], " \t");
        _ = it.next(); // "cpu"
        var ticks: [4]u64 = @splat(0);
        var all: [10]u64 = @splat(0);
        var i: usize = 0;
        while (it.next()) |tok| : (i += 1) {
            if (i >= 10) break;
            all[i] = std.fmt.parseInt(u64, tok, 10) catch 0;
        }
        ticks[0] = all[0] + all[1] + all[2]; // user+nice+system
        ticks[1] = 0;
        ticks[2] = 0;
        ticks[3] = all[3] + all[4]; // idle + iowait
        var total: u64 = 0;
        var idle: u64 = 0;
        for (0..4) |j| {
            const delta = ticks[j] -| prev_ticks[j];
            total += delta;
            if (j == 3) idle = delta;
            prev_ticks[j] = ticks[j];
        }
        if (total == 0) return 0;
        return @intCast((total - idle) * 100 / total);
    }
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

var gpu_cores: ?u32 = null;

/// GPU core count (AGXAccelerator `gpu-core-count`), 0 when unknown. Read once.
pub fn gpuCoreCount() u32 {
    if (gpu_cores) |n| return n;
    gpu_cores = blk: {
        if (comptime !is_macos) break :blk 0;
        const matching = IOServiceMatching("AGXAccelerator") orelse break :blk 0;
        var iter: u32 = 0;
        if (IOServiceGetMatchingServices(0, matching, &iter) != 0) break :blk 0;
        defer _ = IOObjectRelease(iter);
        const entry = IOIteratorNext(iter);
        if (entry == 0) break :blk 0;
        defer _ = IOObjectRelease(entry);
        var props: ?*anyopaque = null;
        if (IORegistryEntryCreateCFProperties(entry, &props, null, 0) != 0) break :blk 0;
        defer if (props) |p| CFRelease(p);
        const v = cfDictGet(props, "gpu-core-count") orelse break :blk 0;
        var n: i64 = 0;
        _ = CFNumberGetValue(v, 4, @ptrCast(&n));
        break :blk if (n > 0 and n < 1 << 16) @intCast(n) else 0;
    };
    return gpu_cores.?;
}

fn cfDictGet(dict: ?*const anyopaque, key_name: [*:0]const u8) ?*const anyopaque {
    const key = CFStringCreateWithCString(null, key_name, 0x08000100) orelse return null;
    defer CFRelease(key);
    return CFDictionaryGetValue(dict, key);
}

test "getAppMemFootprintMb returns a plausible nonzero footprint" {
    const fp = getAppMemFootprintMb();
    // The test process itself footprints several MB; a wrong flavor/offset
    // would yield 0 or absurd garbage.
    try std.testing.expect(fp > 0);
    try std.testing.expect(fp < 1024 * 1024); // < 1 TB sanity bound
}

test "gpuCoreCount reads the GPU's core count" {
    if (comptime !is_macos) return error.SkipZigTest;
    const n = gpuCoreCount();
    if (n == 0) return error.SkipZigTest; // a VM exposes no AGX accelerator entry
    try std.testing.expect(n >= 7 and n <= 256);
}

test "underMemoryPressure: warning and critical stop a bulk allocation, normal and unreadable do not" {
    try std.testing.expect(!underMemoryPressure(0));
    try std.testing.expect(!underMemoryPressure(1));
    try std.testing.expect(underMemoryPressure(2));
    try std.testing.expect(underMemoryPressure(4));
}

test "memoryPressureLevel reads the kernel's level on macOS" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const level = memoryPressureLevel();
    try std.testing.expect(level == 1 or level == 2 or level == 4);
}
