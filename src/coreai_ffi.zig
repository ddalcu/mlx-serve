//! Run-time binding of `lib/coreai/coreai_bridge.h`. The bridge links Core AI (macOS 27), so it is
//! dlopened on first use instead of linked: on an older macOS the binary still starts and an ANE
//! bundle reports `CoreAiUnavailable` by name.

const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");

pub const DType = enum(i32) { f16 = 0, f32 = 1, i32 = 2, i8 = 3, u8 = 4 };

pub const Compute = enum(i32) { ane = 0, gpu = 1, cpu = 2, default = 3 };

pub const ABI_VERSION: i32 = 1;

const Names = [*]const [*:0]const u8;
const Handles = [*]const ?*anyopaque;

pub const Api = struct {
    abi_version: *const fn () callconv(.c) i32,
    release: *const fn (?*anyopaque) callconv(.c) void,
    model_load: *const fn ([*:0]const u8, i32, i32, [*]u8, i32) callconv(.c) ?*anyopaque,
    function_load: *const fn (?*anyopaque, [*:0]const u8, [*]u8, i32) callconv(.c) ?*anyopaque,
    function_describe: *const fn (?*anyopaque, ?[*]u8, i32) callconv(.c) i32,
    buffer_create: *const fn (i32, i32, [*]const i64, ?[*]const i64, [*]u8, i32) callconv(.c) ?*anyopaque,
    buffer_address: *const fn (?*anyopaque) callconv(.c) ?*anyopaque,
    buffer_nbytes: *const fn (?*anyopaque) callconv(.c) i64,
    binding_create: *const fn (?*anyopaque, i32, Names, Handles, i32, Names, Handles, i32, ?Names, ?Handles, [*]u8, i32) callconv(.c) ?*anyopaque,
    run: *const fn (i32, Handles, ?[*]f64, [*]u8, i32) callconv(.c) i32,
};

pub const LIB_NAME = "libcoreai_bridge.dylib";
pub const ENV_PATH = "MLX_SERVE_COREAI_BRIDGE";

/// Where the dylib may sit relative to the executable's directory: beside it (CLI tarball),
/// `zig-out/lib` from `zig-out/bin`, and the app bundle's Frameworks from Contents/MacOS.
pub const exe_relative = [_][]const u8{ LIB_NAME, "../lib/" ++ LIB_NAME, "../Frameworks/" ++ LIB_NAME };

var api: ?Api = null;
var failed = false;
var mu: std.atomic.Mutex = .unlocked;

/// The bridge, loaded once. Fails with `CoreAiUnavailable` (logged once with the reason) when the
/// dylib is missing or does not load (no Core AI on this macOS).
pub fn get() error{CoreAiUnavailable}!*const Api {
    while (!mu.tryLock()) std.atomic.spinLoopHint();
    defer mu.unlock();
    if (api) |*a| return a;
    if (failed) return error.CoreAiUnavailable;
    api = open() catch |err| {
        failed = true;
        log.err("[coreai] bridge unavailable ({s}): needs macOS 27 and {s} beside mlx-serve, or {s}=<path>\n", .{ @errorName(err), LIB_NAME, ENV_PATH });
        return error.CoreAiUnavailable;
    };
    return &api.?;
}

fn open() !Api {
    if (builtin.os.tag != .macos) return error.NotMacos;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (std.c.getenv(ENV_PATH)) |p| return openAt(p);
    const exe = exeDir(&buf) orelse return error.NoExecutablePath;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    for (exe_relative) |rel| {
        const p = std.fmt.bufPrintSentinel(&path_buf, "{s}/{s}", .{ exe, rel }, 0) catch continue;
        if (openAt(p)) |a| return a else |_| {}
    }
    return error.BridgeNotFound;
}

fn openAt(path: [*:0]const u8) !Api {
    const h = std.c.dlopen(path, .{ .NOW = true, .LOCAL = true }) orelse {
        if (std.c.dlerror()) |e| log.debug("[coreai] dlopen {s}: {s}\n", .{ path, e });
        return error.DlopenFailed;
    };
    var a: Api = undefined;
    inline for (@typeInfo(Api).@"struct".field_names) |name| {
        const sym = std.c.dlsym(h, "cai_" ++ name) orelse return error.MissingSymbol;
        @field(a, name) = @ptrCast(@alignCast(sym));
    }
    if (a.abi_version() != ABI_VERSION) return error.AbiMismatch;
    log.info("[coreai] bridge loaded: {s}\n", .{path});
    return a;
}

/// The executable's file name, as Core AI keys its compile cache (`_` becomes `-`).
pub fn exeCacheName(buf: []u8) ?[]const u8 {
    const dir = exeDir(buf) orelse return null;
    const name = buf[dir.len + 1 .. std.mem.indexOfScalar(u8, buf, 0) orelse return null];
    std.mem.replaceScalar(u8, name, '_', '-');
    return name;
}

fn exeDir(buf: []u8) ?[]const u8 {
    switch (builtin.os.tag) {
        .macos => {
            var n: u32 = @intCast(buf.len);
            if (std.c._NSGetExecutablePath(buf.ptr, &n) != 0) return null;
            return std.fs.path.dirname(std.mem.sliceTo(@as([*:0]const u8, @ptrCast(buf.ptr)), 0));
        },
        else => return null,
    }
}

/// An error message buffer for one bridge call.
pub const ErrBuf = struct {
    buf: [1024]u8 = @splat(0),

    pub fn ptr(self: *ErrBuf) [*]u8 {
        self.buf[0] = 0;
        return &self.buf;
    }
    pub fn cap(_: *const ErrBuf) i32 {
        return 1024;
    }
    pub fn text(self: *const ErrBuf) []const u8 {
        return std.mem.sliceTo(&self.buf, 0);
    }
};
