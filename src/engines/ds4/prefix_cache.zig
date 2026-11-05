//! Native ds4 SSD checkpoints. The owning inference thread serializes every call.
//! Main stores once after a successful cold prefill, then before destructive replacement/close;
//! warm tool turns stay in the live session. Store requires the exact live token history.
//! Budget is one root-wide cap; other engines' actual files reduce each write allowance.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const disk_budget = @import("../../sdk/kv/disk_budget.zig");

pub const Identity = struct {
    /// Runtime/build identity, including native payload implementation changes.
    build: []const u8,
    /// The effective rendered-template source, not merely its filename.
    template: []const u8,
    /// Effective engine settings, including context, prefill, MTP and DSpark.
    settings: []const u8,
    /// Every loaded support/projector/settings file; identity includes realpath, size and mtime.
    sidecars: []const []const u8 = &.{},
};

pub const Options = struct {
    home: []const u8,
    /// Single root-wide disk cap, including all other engines' caches.
    budget_bytes: u64,
    identity: Identity,
    /// Zero uses the native default (512); it does not disable the minimum.
    min_tokens: u31 = 0,
};

pub const Reason = enum {
    cold,
    replacement,
    close,

    fn native(self: Reason) [*:0]const u8 {
        return switch (self) {
            .cold => "cold",
            .replacement => "evict",
            .close => "shutdown",
        };
    }
};

pub const Restore = struct {
    /// Owned exact restored prefix. The target prompt itself is unchanged.
    tokens: []i32,
    cached_tokens: usize,
    allocation: []i32,

    pub fn deinit(self: Restore, gpa: std.mem.Allocator) void {
        gpa.free(self.allocation);
    }
};

const Native = opaque {};
extern fn mlxserve_ds4_kv_cache_open(engine: *anyopaque, dir: [*:0]const u8, namespace_key: [*:0]const u8, budget_bytes: u64, min_tokens: c_int, err: [*]u8, errlen: usize) ?*Native;
extern fn mlxserve_ds4_kv_cache_close(cache: *Native) void;
extern fn mlxserve_ds4_kv_cache_store(cache: *Native, session: *anyopaque, tokens: [*]const i32, count: c_int, reason: [*:0]const u8, err: [*]u8, errlen: usize) c_int;
extern fn mlxserve_ds4_kv_cache_restore(cache: *Native, session: *anyopaque, target: [*]const i32, count: c_int, err: [*]u8, errlen: usize) c_int;
extern fn mlxserve_ds4_kv_cache_physical_bytes(cache: *Native) u64;
extern fn mlxserve_ds4_kv_cache_set_budget(cache: *Native, bytes: u64, err: [*]u8, errlen: usize) c_int;
extern fn mlxserve_ds4_kv_cache_prune(cache: *Native, err: [*]u8, errlen: usize) c_int;

/// `B` is bridge.zig; only its engine/session handle fields cross the C boundary.
pub fn Cache(comptime B: type) type {
    return struct {
        const Self = @This();
        allocator: std.mem.Allocator,
        io: std.Io,
        handle: *Native,
        base: []u8,
        root: [:0]u8,
        total_cap: u64,

        pub fn open(io: std.Io, gpa: std.mem.Allocator, engine: *B.Ds4Engine, model_path: []const u8, options: Options) !Self {
            const ns = try namespace(io, gpa, model_path, options.identity);
            defer gpa.free(ns);
            const dir = try poolDir(gpa, options.home);
            errdefer gpa.free(dir);
            const base = try std.fmt.allocPrint(gpa, "{s}/.mlx-serve/kv-cache", .{options.home});
            errdefer gpa.free(base);
            const allowance = try disk_budget.availableBudget(gpa, io, base, dir, options.budget_bytes, 0);
            var err: [256]u8 = @splat(0);
            const handle = mlxserve_ds4_kv_cache_open(@ptrCast(engine.handle), dir, ns, allowance orelse 0, options.min_tokens, &err, err.len) orelse {
                report("open", &err);
                return error.Ds4PrefixCacheOpen;
            };
            return .{ .allocator = gpa, .io = io, .handle = handle, .base = base, .root = dir, .total_cap = options.budget_bytes };
        }

        /// Flush the session with store(.close) before deinit; the cache never owns it.
        pub fn deinit(self: *Self) void {
            mlxserve_ds4_kv_cache_close(self.handle);
            self.allocator.free(self.base);
            self.allocator.free(self.root);
            self.* = undefined;
        }

        /// Returns false for ineligible/under-minimum/over-budget checkpoints. The native
        /// boundary rejects invalid (including interrupted) and image-conditioned sessions.
        pub fn store(self: *Self, session: *B.Ds4Session, prompt: []const i32, reason: Reason) !bool {
            if (prompt.len > std.math.maxInt(c_int)) return error.Ds4PrefixCachePromptTooLong;
            const reservation = (try disk_budget.reserve(self.allocator, self.io, self.base, self.root, self.total_cap, null)) orelse {
                try self.applyAllowance(0);
                return false;
            };
            defer reservation.release(self.io);
            try self.applyAllowance(reservation.bytes);
            var err: [256]u8 = @splat(0);
            const rc = mlxserve_ds4_kv_cache_store(self.handle, @ptrCast(session.handle), prompt.ptr, @intCast(prompt.len), reason.native(), &err, err.len);
            if (rc < 0) {
                report("store", &err);
                return error.Ds4PrefixCacheStore;
            }
            return rc > 0;
        }

        /// Native load passes effective_prompt=null and then verifies the actual session
        /// tokens against target. A failed payload load invalidates any mutated session.
        pub fn restore(self: *Self, session: *B.Ds4Session, target: []const i32) !?Restore {
            if (target.len > std.math.maxInt(c_int)) return error.Ds4PrefixCachePromptTooLong;
            // Allocate before the native mutation so allocation failure cannot hide a loaded state.
            const tokens = try self.allocator.dupe(i32, target);
            errdefer self.allocator.free(tokens);
            var err: [256]u8 = @splat(0);
            const rc = mlxserve_ds4_kv_cache_restore(self.handle, @ptrCast(session.handle), target.ptr, @intCast(target.len), &err, err.len);
            if (rc < 0) {
                report("restore", &err);
                return error.Ds4PrefixCacheRestore;
            }
            if (rc == 0) {
                self.allocator.free(tokens);
                return null;
            }
            const n: usize = @intCast(rc);
            return .{ .tokens = tokens[0..n], .cached_tokens = n, .allocation = tokens };
        }

        /// Actual file lengths, including unindexed/corrupt files. maxInt(u64) on scan
        /// failure is conservative when main subtracts this from the total cache budget.
        pub fn physicalDiskBytes(self: *Self) u64 {
            return mlxserve_ds4_kv_cache_physical_bytes(self.handle);
        }

        /// Update the single root-wide cap. Zero stops writes while retaining
        /// read-only restores; call prune explicitly to remove those checkpoints.
        pub fn setBudget(self: *Self, bytes: u64) !void {
            self.total_cap = bytes;
            try self.refreshBudget();
        }

        fn applyAllowance(self: *Self, bytes: u64) !void {
            var err: [256]u8 = @splat(0);
            if (mlxserve_ds4_kv_cache_set_budget(self.handle, bytes, &err, err.len) != 0) {
                report("budget", &err);
                return error.Ds4PrefixCacheBudget;
            }
        }

        /// Open and store refresh automatically; no other engine's files are deleted here.
        pub fn refreshBudget(self: *Self) !void {
            const allowance = disk_budget.availableBudget(self.allocator, self.io, self.base, self.root, self.total_cap, 0) catch |err| {
                self.applyAllowance(0) catch |budget_err| {
                    log.warn("[ds4-cache] disabling after budget scan failure: {t}\n", .{budget_err});
                };
                return err;
            };
            try self.applyAllowance(allowance orelse 0);
        }

        pub fn prune(self: *Self) !void {
            var err: [256]u8 = @splat(0);
            if (mlxserve_ds4_kv_cache_prune(self.handle, &err, err.len) != 0) {
                report("prune", &err);
                return error.Ds4PrefixCacheBudget;
            }
        }
    };
}

fn report(operation: []const u8, err: []const u8) void {
    log.warn("[ds4-cache] {s}: {s}\n", .{ operation, std.mem.sliceTo(err, 0) });
}

pub fn poolDir(gpa: std.mem.Allocator, home: []const u8) ![:0]u8 {
    if (!std.fs.path.isAbsolute(home)) return error.Ds4PrefixCacheHome;
    return std.fmt.allocPrintSentinel(gpa, "{s}/.mlx-serve/kv-cache/ds4", .{home}, 0);
}

/// Length-delimited fields prevent ambiguous concatenation of settings and template.
fn field(hash: *std.crypto.hash.sha2.Sha256, bytes: []const u8) void {
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, bytes.len, .little);
    hash.update(&length);
    hash.update(bytes);
}

fn fileIdentity(io: std.Io, gpa: std.mem.Allocator, hash: *std.crypto.hash.sha2.Sha256, path: []const u8) !void {
    const resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, path, gpa);
    defer gpa.free(resolved);
    const file = try std.Io.Dir.openFileAbsolute(io, resolved, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.Ds4PrefixCacheModelFile;
    field(hash, resolved);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, stat.size, .little);
    hash.update(&size);
    var mtime: [16]u8 = undefined;
    std.mem.writeInt(i128, &mtime, stat.mtime.nanoseconds, .little);
    hash.update(&mtime);
}

pub fn namespace(io: std.Io, gpa: std.mem.Allocator, model_path: []const u8, identity: Identity) ![:0]u8 {
    if (identity.build.len == 0) return error.Ds4PrefixCacheBuildIdentity;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    field(&hash, "mlx-serve/ds4/token-prefix/v1");
    try fileIdentity(io, gpa, &hash, model_path);
    field(&hash, identity.build);
    field(&hash, identity.template);
    field(&hash, identity.settings);
    for (identity.sidecars) |path| try fileIdentity(io, gpa, &hash, path);
    const digest = hash.finalResult();
    return gpa.dupeSentinel(u8, &std.fmt.bytesToHex(digest, .lower), 0);
}

test "ds4 namespace separates checkpoint settings template build and sidecars" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = "weights" });
    try tmp.dir.writeFile(io, .{ .sub_path = "support.gguf", .data = "draft" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const model = try std.fmt.allocPrint(gpa, "{s}/model.gguf", .{root});
    defer gpa.free(model);
    const sidecar = try std.fmt.allocPrint(gpa, "{s}/support.gguf", .{root});
    defer gpa.free(sidecar);
    const initial: Identity = .{ .build = "build", .template = "template", .settings = "ctx=8192" };
    const baseline = try namespace(io, gpa, model, initial);
    defer gpa.free(baseline);
    const variants = [_]Identity{
        .{ .build = "other", .template = initial.template, .settings = initial.settings },
        .{ .build = initial.build, .template = "other", .settings = initial.settings },
        .{ .build = initial.build, .template = initial.template, .settings = "ctx=4096" },
        .{ .build = initial.build, .template = initial.template, .settings = initial.settings, .sidecars = &.{sidecar} },
    };
    for (variants) |variant| {
        const changed = try namespace(io, gpa, model, variant);
        defer gpa.free(changed);
        try std.testing.expect(!std.mem.eql(u8, baseline, changed));
    }
    const with_sidecar = variants[3];
    const sidecar_before = try namespace(io, gpa, model, with_sidecar);
    defer gpa.free(sidecar_before);
    try tmp.dir.writeFile(io, .{ .sub_path = "support.gguf", .data = "changed draft" });
    const sidecar_after = try namespace(io, gpa, model, with_sidecar);
    defer gpa.free(sidecar_after);
    try std.testing.expect(!std.mem.eql(u8, sidecar_before, sidecar_after));
    const model_file = try tmp.dir.openFile(io, "model.gguf", .{});
    defer model_file.close(io);
    const stat = try model_file.stat(io);
    try model_file.setTimestamps(io, .{ .modify_timestamp = .{ .new = .{ .nanoseconds = stat.mtime.nanoseconds - std.time.ns_per_s } } });
    const retimed = try namespace(io, gpa, model, initial);
    defer gpa.free(retimed);
    try std.testing.expect(!std.mem.eql(u8, baseline, retimed));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = "new weights" });
    const changed = try namespace(io, gpa, model, initial);
    defer gpa.free(changed);
    try std.testing.expect(!std.mem.eql(u8, baseline, changed));
    try std.testing.expectError(error.Ds4PrefixCacheBuildIdentity, namespace(io, gpa, model, .{ .build = "", .template = "", .settings = "" }));
}

test "ds4 namespace resolves aliases and separates same-name checkpoints" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "other");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.gguf", .data = "weights" });
    try tmp.dir.writeFile(io, .{ .sub_path = "other/model.gguf", .data = "weights" });
    try tmp.dir.symLink(io, "model.gguf", "alias.gguf", .{});
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const id: Identity = .{ .build = "build", .template = "ab", .settings = "c" };
    var values: [3][:0]u8 = undefined;
    for ([_][]const u8{ "model.gguf", "alias.gguf", "other/model.gguf" }, 0..) |name, i| {
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, name });
        defer gpa.free(path);
        values[i] = try namespace(io, gpa, path, id);
    }
    defer for (values) |value| gpa.free(value);
    try std.testing.expectEqualStrings(values[0], values[1]);
    try std.testing.expect(!std.mem.eql(u8, values[0], values[2]));
    const path = try std.fmt.allocPrint(gpa, "{s}/model.gguf", .{root});
    defer gpa.free(path);
    const delimited = try namespace(io, gpa, path, .{ .build = "build", .template = "a", .settings = "bc" });
    defer gpa.free(delimited);
    try std.testing.expect(!std.mem.eql(u8, values[0], delimited));
}
