//! A loaded media plugin on `LoadedModel.media_plugin`: its table, its handle and the pack it read, plus the bills and
//! the machine facts the table's bill functions take. Load, job and unload run on the inference thread.

const std = @import("std");
const abi = @import("../../../abi/media.zig");
const mlx = @import("../../../mlx/ffi.zig");
const w8a8 = @import("../../../sdk/blocks/dit/w8a8.zig");
const ane = @import("../../../sdk/ane/offload.zig");
const log = @import("../../../sdk/util/log.zig");
const served = @import("served.zig");
const low_mem = @import("../low_mem.zig");
const status = @import("../../status.zig");
const gen = @import("../gen.zig");
const FsPackDir = @import("../../registry/pack_dir.zig").FsPackDir;

pub const MediaTable = abi.MediaTable;

/// The launch and machine facts a bill reads.
pub fn machine() abi.Machine {
    return .{
        .gpu_working_set = mlx.maxRecommendedWorkingSet(),
        .w8a8 = w8a8.w8a8State().enabled and w8a8.nax(),
        .ane_image = ane.aneState().media_offload.image,
        .ane_video = ane.aneState().media_offload.video,
        .ane_audio = ane.aneState().media_offload.audio,
        .low_mem = low_mem.default(),
        .ram_total = status.getTotalMemBytes(),
        .ram_available = status.getAvailableMemBytes(),
    };
}

/// The plugin's own name for the last load it refused on this thread (an `@errorName`, static).
threadlocal var refused_load: ?[]const u8 = null;

/// The name a failed load reports (registry `error_name`, the named 503): the plugin's own for a refused plugin load.
pub fn loadErrorName(err: anyerror) []const u8 {
    if (err == error.MediaPluginLoadFailed) if (refused_load) |n| return n;
    return @errorName(err);
}

/// `t`'s load peak for the pack in `dir`.
pub fn billLoad(io: std.Io, dir: std.Io.Dir, t: *const MediaTable) u64 {
    const fs: FsPackDir = .{ .io = io, .root = dir };
    const m = machine();
    return t.bill_load(&fs.packDir(), &m);
}

/// `t`'s caps for the pack in `dir`.
pub fn caps(io: std.Io, dir: std.Io.Dir, t: *const MediaTable) abi.MediaCaps {
    const fs: FsPackDir = .{ .io = io, .root = dir };
    var out: abi.MediaCaps = .{};
    t.caps(&fs.packDir(), &out);
    return out;
}

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    table: *const MediaTable,
    handle: *anyopaque,
    /// The pack's folder, open for the plugin's lifetime: request bills and jobs read it (`pack`).
    dir: std.Io.Dir,
    fs: FsPackDir,
    pack: abi.PackDir,
    caps: abi.MediaCaps = .{},
    /// Read at load, on the inference thread: request bills reuse it off that thread.
    machine: abi.Machine = .{},
    /// Decision plugins: how long a pass waits for more requests to merge.
    batch_window_us: u32 = 0,

    /// Loads the pack at `model_dir` through `t` on stream `s`; a refusal is logged with the plugin's reason.
    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, t: *const MediaTable, s: mlx.mlx_stream) !*Engine {
        if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.FileNotFound; // openDirAbsolute UB class
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        errdefer dir.close(io);
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        const fs: FsPackDir = .{ .io = io, .root = dir };
        const pd = fs.packDir();
        const m = machine();
        const window = if (t.modality == .decision) gen.decisionBatchWindowUs() else 0;
        var settings_buf: [128]u8 = undefined;
        const settings = if (t.modality == .decision) gen.decisionSettingsJson(&settings_buf, window) else "";
        var why: abi.Verdict = .{};
        const handle = t.load(&.{
            .model_dir = .of(model_dir),
            .dir = &pd,
            .settings_json = .of(settings),
            .stream = .{ .ctx = s.ctx },
            .machine = &m,
        }, &why) orelse {
            log.err("[gen] {s} does not load: {s}\n", .{ t.name.slice(), why.why.slice() });
            if (why.code == .busy) return error.OutOfMemory;
            refused_load = why.why.slice();
            return error.MediaPluginLoadFailed;
        };
        self.* = .{ .allocator = allocator, .io = io, .table = t, .handle = handle, .dir = dir, .fs = fs, .pack = undefined, .machine = m, .batch_window_us = window };
        self.pack = self.fs.packDir();
        t.caps(&pd, &self.caps);
        log.info("[gen] {s} ready through its plugin table\n", .{t.name.slice()});
        if (t.modality == .decision) log.info("[decision] {s} engine ready\n", .{t.name.slice()});
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.table.unload(self.handle);
        self.dir.close(self.io);
        self.allocator.destroy(self);
    }

    /// The scheduler's reserve for one request (`GenRequest.peak_bytes`): the request's whole peak, or 0 (the load
    /// bill covers it).
    pub fn requestPeak(self: *const Engine, body: []const u8) u64 {
        return self.table.bill_request(&self.pack, &self.machine, .of(body));
    }

    /// Whether `e` (a model's plugin slot) serves modality `m` (the host's `Modality`).
    pub fn serves(e: ?*const Engine, m: anytype) bool {
        const p = e orelse return false;
        return served.modalityAs(@TypeOf(m), p.table) == m;
    }

    /// Whether `e` serves the audio route named `route` ("speech", "music" or "sound").
    pub fn servesRoute(e: ?*const Engine, comptime route: []const u8) bool {
        const p = e orelse return false;
        return p.table.modality == .audio and @field(p.caps.routes, route);
    }
};
