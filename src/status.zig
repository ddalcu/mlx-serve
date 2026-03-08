const std = @import("std");

pub const State = enum { idle, prefill, decode };

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

// ── Background refresh thread state ──
var bg_state: u8 = 0;
var bg_running: bool = false;
var bg_rows: u16 = 0;
var bg_cols: u16 = 0;
var bg_thread: ?std.Thread = null;

fn refreshLoop() void {
    while (@atomicLoad(bool, &bg_running, .monotonic)) {
        std.Thread.sleep(5 * std.time.ns_per_s);
        if (!@atomicLoad(bool, &bg_running, .monotonic)) break;
        const state: State = @enumFromInt(@atomicLoad(u8, &bg_state, .monotonic));
        const sb = StatusBar{ .rows = bg_rows, .cols = bg_cols, .enabled = true };
        sb.update(state, null);
    }
}

// ── Metric helpers ──

fn getAppRssMb() u32 {
    var info = std.mem.zeroes(TaskBasicInfo);
    var count: u32 = @sizeOf(TaskBasicInfo) / @sizeOf(i32);
    if (task_info(mach_task_self_, 20, @ptrCast(&info), &count) != 0) return 0;
    return @intCast(info.resident_size / (1024 * 1024));
}

fn getSysMemPct() u32 {
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

fn getCpuPct() u32 {
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

fn getGpuPct() u32 {
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

// ── StatusBar ──

pub const StatusBar = struct {
    rows: u16,
    cols: u16,
    enabled: bool,

    pub fn init() StatusBar {
        if (isatty(2) == 0)
            return .{ .rows = 0, .cols = 0, .enabled = false };
        const size = getTermSize();
        const self = StatusBar{ .rows = size.rows, .cols = size.cols, .enabled = true };
        var buf: [32]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, "\x1b[1;{d}r\x1b[1;1H", .{self.rows - 1}) catch return self;
        std.fs.File.stderr().writeAll(seq) catch {};
        self.update(.idle, null);

        bg_rows = self.rows;
        bg_cols = self.cols;
        @atomicStore(bool, &bg_running, true, .monotonic);
        bg_thread = std.Thread.spawn(.{}, refreshLoop, .{}) catch null;

        return self;
    }

    pub fn update(self: StatusBar, state: State, detail: ?[]const u8) void {
        if (!self.enabled) return;
        @atomicStore(u8, &bg_state, @intFromEnum(state), .monotonic);
        var buf: [1024]u8 = undefined;
        const prefix = std.fmt.bufPrint(&buf, "\x1b7\x1b[{d};1H\x1b[2K", .{self.rows}) catch return;
        var pos: usize = prefix.len;

        const style: []const u8 = switch (state) {
            .idle => "\x1b[48;5;28m\x1b[97m",
            .prefill => "\x1b[48;5;172m\x1b[97m",
            .decode => "\x1b[48;5;25m\x1b[97m",
        };
        @memcpy(buf[pos..][0..style.len], style);
        pos += style.len;

        const label: []const u8 = switch (state) {
            .idle => "  IDLE  ",
            .prefill => "  PREFILL  ",
            .decode => "  DECODE  ",
        };
        @memcpy(buf[pos..][0..label.len], label);
        pos += label.len;
        var used: usize = label.len;

        if (detail) |d| {
            const n = @min(d.len, @as(usize, self.cols) -| used -| 1);
            @memcpy(buf[pos..][0..n], d[0..n]);
            pos += n;
            used += n;
        }

        // Build right-aligned metrics
        var rss_buf: [16]u8 = undefined;
        const rss = getAppRssMb();
        const rss_str = if (rss >= 1024)
            std.fmt.bufPrint(&rss_buf, "{d}.{d}G", .{ rss / 1024, (rss % 1024) * 10 / 1024 }) catch return
        else
            std.fmt.bufPrint(&rss_buf, "{d}M", .{rss}) catch return;

        var metrics_buf: [80]u8 = undefined;
        const metrics = std.fmt.bufPrint(&metrics_buf, "{s}  Mem {d}%  CPU {d}%  GPU {d}% ", .{
            rss_str, getSysMemPct(), getCpuPct(), getGpuPct(),
        }) catch return;

        if (used + metrics.len < self.cols) {
            const gap = @as(usize, self.cols) - used - metrics.len;
            var gi: usize = 0;
            while (gi < gap and pos + metrics.len + 16 < buf.len) : (gi += 1) {
                buf[pos] = ' ';
                pos += 1;
            }
            @memcpy(buf[pos..][0..metrics.len], metrics);
            pos += metrics.len;
        } else {
            while (used < self.cols and pos < buf.len - 16) : (used += 1) {
                buf[pos] = ' ';
                pos += 1;
            }
        }

        const suffix = "\x1b[0m\x1b8";
        @memcpy(buf[pos..][0..suffix.len], suffix);
        pos += suffix.len;

        std.fs.File.stderr().writeAll(buf[0..pos]) catch {};
    }

    pub fn deinit(self: StatusBar) void {
        @atomicStore(bool, &bg_running, false, .monotonic);
        if (bg_thread) |t| t.detach();
        bg_thread = null;

        if (!self.enabled) return;
        var buf: [64]u8 = undefined;
        const seq = std.fmt.bufPrint(&buf, "\x1b[r\x1b[{d};1H\x1b[2K", .{self.rows}) catch return;
        std.fs.File.stderr().writeAll(seq) catch {};
    }
};

const Winsize = extern struct { ws_row: u16, ws_col: u16, ws_xpixel: u16, ws_ypixel: u16 };
const TIOCGWINSZ: c_ulong = 0x40087468;

extern "c" fn ioctl(fd: c_int, request: c_ulong, ...) c_int;
extern "c" fn isatty(fd: c_int) c_int;

fn getTermSize() struct { rows: u16, cols: u16 } {
    var ws = std.mem.zeroes(Winsize);
    if (ioctl(2, TIOCGWINSZ, &ws) == 0 and ws.ws_row > 0)
        return .{ .rows = ws.ws_row, .cols = ws.ws_col };
    return .{ .rows = 24, .cols = 80 };
}
