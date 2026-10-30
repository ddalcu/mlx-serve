//! Process and box memory readings from the kernel's own ledgers (task_vm_info rev3, rusage v4, vm_statistics64):
//! read-only probes a plugin's memory bill and construction checks compare against. Zero where the platform has no
//! ledger. The host's status bar reads the same functions (status.zig re-exports them).

const std = @import("std");
const builtin = @import("builtin");

extern "c" var mach_task_self_: u32;
extern "c" fn mach_host_self() u32;
extern "c" fn task_info(task: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_statistics64(host: u32, flavor: u32, info: [*]i32, cnt: *u32) i32;
extern "c" fn host_page_size(host: u32, out: *usize) i32;
extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// task_vm_info through the rev3 ledger block. Field order matches <mach/task_info.h>
/// exactly; @sizeOf(TaskVmInfo)/@sizeOf(i32) == 84 == TASK_VM_INFO_REV3_COUNT. The count
/// is in/out: a kernel below rev3 fills fewer fields and leaves the rest zero.
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
    // rev2
    min_address: u64,
    max_address: u64,
    // rev3: the ledger block
    ledger_phys_footprint_peak: i64,
    ledger_purgeable_nonvolatile: i64,
    ledger_purgeable_novolatile_compressed: i64,
    ledger_purgeable_volatile: i64,
    ledger_purgeable_volatile_compressed: i64,
    ledger_tag_network_nonvolatile: i64,
    ledger_tag_network_nonvolatile_compressed: i64,
    ledger_tag_network_volatile: i64,
    ledger_tag_network_volatile_compressed: i64,
    ledger_tag_media_footprint: i64,
    ledger_tag_media_footprint_compressed: i64,
    ledger_tag_media_nofootprint: i64,
    ledger_tag_media_nofootprint_compressed: i64,
    ledger_tag_graphics_footprint: i64,
    ledger_tag_graphics_footprint_compressed: i64,
    ledger_tag_graphics_nofootprint: i64,
    ledger_tag_graphics_nofootprint_compressed: i64,
    ledger_tag_neural_footprint: i64,
    ledger_tag_neural_footprint_compressed: i64,
    ledger_tag_neural_nofootprint: i64,
    ledger_tag_neural_nofootprint_compressed: i64,
};

comptime {
    std.debug.assert(@sizeOf(TaskVmInfo) / @sizeOf(i32) == 84); // TASK_VM_INFO_REV3_COUNT
}

/// vm_statistics64 (<mach/vm_statistics.h>), HOST_VM_INFO64.
pub const VmStats64 = extern struct {
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

fn taskVmInfo() ?TaskVmInfo {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    var info = std.mem.zeroes(TaskVmInfo);
    var count: u32 = @sizeOf(TaskVmInfo) / @sizeOf(i32); // 84 = TASK_VM_INFO_REV3_COUNT (in/out)
    if (task_info(mach_task_self_, 22, @ptrCast(&info), &count) != 0) return null;
    return info;
}

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const MemoryState = struct {
    /// Bytes the host's other models hold in this process (its own bill's footprint reads count them too). The host
    /// keeps it current; a plugin's absolute footprint bounds rise by it.
    foreign_bytes: std.atomic.Value(u64) = .init(0),
};
pub const memoryState_shared = @import("mlx_host").shared.Shared(MemoryState, "streamed_memory", .{});
inline fn memoryState() *MemoryState {
    return memoryState_shared.get();
}

pub fn setForeignBytes(n: u64) void {
    memoryState().foreign_bytes.store(n, .monotonic);
}

pub fn foreignBytes() u64 {
    return memoryState().foreign_bytes.load(.monotonic);
}

/// This process's phys_footprint (MLX's Metal / IOKit memory included) and its lifetime peak, in bytes.
pub const Footprint = struct { now: u64, peak: u64 };

pub fn footprint() Footprint {
    const info = taskVmInfo() orelse return .{ .now = 0, .peak = 0 };
    return .{ .now = info.phys_footprint, .peak = if (info.ledger_phys_footprint_peak > 0) @intCast(info.ledger_phys_footprint_peak) else info.phys_footprint };
}

/// rusage_info_v4 (<sys/resource.h>): the footprint's interval high-water mark, which
/// `proc_reset_footprint_interval` (libsystem_kernel) restarts: the kernel's own ledger, no sampling.
const RusageInfoV4 = extern struct {
    uuid: [16]u8,
    f: [35]u64,
    const lifetime_max_phys_footprint = 28;
    const interval_max_phys_footprint = 33;
};
extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *RusageInfoV4) c_int;
extern "c" fn proc_reset_footprint_interval(pid: c_int) c_int;
extern "c" fn getpid() c_int;

/// This process's memory from the kernel's ledgers (task_vm_info rev3 and rusage v4), in bytes: the
/// footprint, its interval and lifetime peaks, and how it splits. Zero where the platform has no ledger.
pub const ProcessMemory = struct {
    /// phys_footprint now (Metal / IOAccelerator included).
    footprint: u64 = 0,
    /// The footprint's high-water mark since the previous `startFootprintInterval`.
    footprint_interval_peak: u64 = 0,
    footprint_lifetime_peak: u64 = 0,
    /// Anonymous pages resident; compressed pages this task owns.
    internal: u64 = 0,
    compressed: u64 = 0,
    /// File-backed pages this task maps resident (outside the footprint).
    external: u64 = 0,
    /// IOKit graphics memory (Metal buffers): in the footprint / outside it.
    graphics_footprint: u64 = 0,
    graphics_nofootprint: u64 = 0,
    /// Volatile purgeable memory (outside the footprint).
    purgeable_volatile: u64 = 0,
};

pub fn processMemory() ProcessMemory {
    const info = taskVmInfo() orelse return .{};
    const pos = struct {
        fn f(x: i64) u64 {
            return if (x > 0) @intCast(x) else 0;
        }
    }.f;
    var m: ProcessMemory = .{
        .footprint = info.phys_footprint,
        .footprint_lifetime_peak = pos(info.ledger_phys_footprint_peak),
        .internal = info.internal,
        .compressed = info.compressed,
        .external = info.external,
        .graphics_footprint = pos(info.ledger_tag_graphics_footprint),
        .graphics_nofootprint = pos(info.ledger_tag_graphics_nofootprint),
        .purgeable_volatile = pos(info.ledger_purgeable_volatile),
    };
    var ru = std.mem.zeroes(RusageInfoV4);
    if (proc_pid_rusage(getpid(), 4, &ru) == 0) {
        m.footprint_interval_peak = ru.f[RusageInfoV4.interval_max_phys_footprint];
        m.footprint_lifetime_peak = @max(m.footprint_lifetime_peak, ru.f[RusageInfoV4.lifetime_max_phys_footprint]);
    }
    return m;
}

/// Restarts the footprint's interval high-water mark (`ProcessMemory.footprint_interval_peak`).
pub fn startFootprintInterval() void {
    if (comptime !builtin.os.tag.isDarwin()) return;
    _ = proc_reset_footprint_interval(getpid());
}

/// The box's page counts (`VmStats64`), in bytes. Zero off Darwin.
pub const VmBytes = struct { free: u64 = 0, active: u64 = 0, inactive: u64 = 0, wired: u64 = 0, purgeable: u64 = 0, speculative: u64 = 0, compressor: u64 = 0, external: u64 = 0, internal: u64 = 0 };

pub fn vmBytes() VmBytes {
    return vmBytesChecked() orelse .{};
}

/// `vmBytes`, or null where the kernel gives no reading (off Darwin, or host_statistics64 refused): a caller whose
/// bound rests on the reading refuses rather than read zeros.
pub fn vmBytesChecked() ?VmBytes {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    var page: usize = 0;
    if (host_page_size(mach_host_self(), &page) != 0) return null;
    var vm = std.mem.zeroes(VmStats64);
    var count: u32 = @sizeOf(VmStats64) / @sizeOf(i32);
    if (host_statistics64(mach_host_self(), 4, @ptrCast(&vm), &count) != 0) return null;
    const pg: u64 = page;
    return .{
        .free = vm.free_count * pg,
        .active = vm.active_count * pg,
        .inactive = vm.inactive_count * pg,
        .wired = vm.wire_count * pg,
        .purgeable = vm.purgeable_count * pg,
        .speculative = vm.speculative_count * pg,
        .compressor = vm.compressor_page_count * pg,
        .external = vm.external_page_count * pg,
        .internal = vm.internal_page_count * pg,
    };
}

/// vm_stat's wired + active + inactive + compressor-occupied pages: the "physical used" an external
/// memory monitor reads (file cache included; free and speculative pages excluded).
pub fn physicalUsedBytes(v: VmBytes) u64 {
    return v.wired + v.active + v.inactive + v.compressor;
}

/// The box's used memory as the host's baseline counts it (wired, the compressor's pages, the anonymous pages less
/// the purgeable ones; file-backed pages aside), from one host_statistics64 reading, whose counts every Darwin has (the
/// `vm.page_wired_count` sysctl an earlier form read does not exist on macOS 26.6: it read 0 there, and a bill took the
/// rest of the box as empty). host_statistics64 is rate-limited box-wide for non-platform binaries (past 2-10 calls a
/// second it returns its last reading): a phase boundary reads it once. Null where the kernel gives no reading; a
/// construction that bills the rest of the box refuses then.
pub fn boxUsedBytes() ?u64 {
    const v = vmBytesChecked() orelse return null;
    return v.wired + (v.internal -| v.purgeable) + v.compressor;
}

/// Total physical RAM (hw.memsize). 0 on failure and off Darwin.
pub fn totalMemBytes() u64 {
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    var total_mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (sysctlbyname("hw.memsize", @ptrCast(&total_mem), &len, null, 0) != 0) return 0;
    return total_mem;
}

const testing = std.testing;

test "sdk memory: the process and box readings are the kernel's, self-consistent, and zero only off Darwin" {
    if (comptime !builtin.os.tag.isDarwin()) {
        try testing.expectEqual(Footprint{ .now = 0, .peak = 0 }, footprint());
        try testing.expectEqual(ProcessMemory{}, processMemory());
        try testing.expectEqual(VmBytes{}, vmBytes());
        try testing.expectEqual(@as(u64, 0), totalMemBytes());
        return;
    }
    const total = totalMemBytes();
    try testing.expect(total >= 1 << 30);
    const f = footprint();
    try testing.expect(f.now > 0 and f.peak >= f.now and f.peak <= total);
    startFootprintInterval();
    // a touched allocation the footprint must count
    const block = try testing.allocator.alloc(u8, 32 << 20);
    defer testing.allocator.free(block);
    @memset(block, 0xa5);
    const m = processMemory();
    try testing.expect(m.footprint > 0 and m.footprint_lifetime_peak >= m.footprint);
    try testing.expect(m.footprint_interval_peak >= 32 << 20 and m.footprint_interval_peak <= m.footprint_lifetime_peak);
    const v = vmBytes();
    try testing.expect(v.wired > 0 and v.active > 0);
    try testing.expect(physicalUsedBytes(v) <= total and physicalUsedBytes(v) >= v.wired);
    // The box's used memory holds the wired pages and this process's anonymous ones, and fits the box.
    const used = boxUsedBytes().?;
    try testing.expect(used <= total and used >= v.wired / 2 and used >= 32 << 20);
}

test "sdk memory: physical used is wired + active + inactive + compressor, free and speculative and file counts aside" {
    const v: VmBytes = .{ .free = 1000, .active = 1, .inactive = 20, .wired = 300, .purgeable = 5000, .speculative = 7000, .compressor = 4000, .external = 50000, .internal = 600000 };
    try testing.expectEqual(@as(u64, 4321), physicalUsedBytes(v));
}
