//! sushi (https://github.com/beamivalice/sushi) as a guest engine: the pinned
//! prebuilt release, run as a separate process on loopback. It serves the
//! Qwen3.8-Flash-Next EXL3 packs. mlx-serve starts it when such a pack loads,
//! forwards the inference routes to it unchanged, and stops it on unload.

const std = @import("std");
const build_options = @import("build_options");
const lan = @import("../lan.zig");
const log = @import("../log.zig");
const io_util = @import("../io_util.zig");

pub const engine_name = "sushi";
/// Pinned by scripts/fetch-sushi.sh (`-Dsushi-tag` + `-Dsushi-sha256` override both).
pub const release_tag = build_options.sushi_tag;
pub const release_sha256 = build_options.sushi_sha256;
/// Digest of the staged release tree (`treeSha256Hex`), re-checked before every
/// launch: the binary, its dylibs and the metallib all run.
pub const release_tree_sha256 = build_options.sushi_tree_sha256;
/// The guest contract this host speaks: `guest_api` in `sushi --guest-manifest`.
pub const guest_api: i64 = 1;
/// The App Store build cannot run a downloaded binary; Linux and iOS have no release.
pub const supported = build_options.macos_engines and !build_options.mas;

const release_asset = "sushi-bin-macos-arm64.tar.gz";
const release_dir = "sushi-macos-arm64";
/// A guest that is still alive is still loading; a cold 64 GB pack takes minutes.
const ready_timeout_ms: i64 = 15 * 60 * 1000;
const stop_grace_ms: i64 = 10 * 1000;
const fetch_timeout_ms: i64 = 10 * 1000;
/// The guest logs to stderr only, into a file the host rotates at each start.
const log_max_bytes: i64 = 32 * 1024 * 1024;
/// The guest requires this per-launch key even from loopback (`--api-key-strict`):
/// its port is open to every local user. Passed in the environment, not argv (`ps`).
const key_env = "MLX_SERVE_SUSHI_KEY";

pub const Options = struct {
    /// `--sushi-path`: run this binary instead of the pinned release.
    path: ?[]const u8 = null,
    /// `--no-sushi`: never download or start the guest.
    disabled: bool = false,
    /// This server's port. The guest listens on the first free port above it
    /// and logs to `~/.mlx-serve/logs/sushi-<port>.log`.
    host_port: u16 = 11234,
};

/// Set once by `main` before the first load.
pub var options: Options = .{};
/// Set by the server's signal handler: a download or a guest still loading gives up.
pub var stopping = std.atomic.Value(bool).init(false);
/// One download at a time: loads for the pack can race to fetch it.
var release_mutex: std.Io.Mutex = .init;

/// Failures that say nothing about the pack: the entry stays loadable and the
/// next request retries.
pub const TransientError = error{ SushiDownloadFailed, SushiNoFreePort, SushiGuestTimeout, SushiGuestUnreachable };

pub fn transientFromName(name: []const u8) ?TransientError {
    inline for (@typeInfo(TransientError).error_set.error_names.?) |e| {
        if (std.mem.eql(u8, name, e)) return @field(TransientError, e);
    }
    return null;
}

pub fn isTransient(err: anyerror) bool {
    return transientFromName(@errorName(err)) != null;
}

/// The binary to run, fetched and checked before anything is evicted for it:
/// a pack whose engine cannot start never unloads another model. Caller frees.
pub fn prepare(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    if (options.disabled) return error.SushiDisabled;
    if (!supported) return error.SushiUnavailable;
    release_mutex.lockUncancelable(io);
    defer release_mutex.unlock(io);
    const bin = if (options.path) |p| try allocator.dupe(u8, p) else try ensureRelease(allocator, io, homeDir());
    errdefer allocator.free(bin);
    try verifyGuest(allocator, io, bin);
    return bin;
}

pub const SushiGuest = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pid: std.posix.pid_t,
    port: u16,
    /// Hex of this launch's random API key.
    key: [64]u8,
    /// The guest's own `/v1/models` row for the model it serves.
    row: Row = .{},
    /// Serialises reaping against signalling, so a reaped pid is never signalled.
    mutex: std.Io.Mutex = .init,
    reaped: bool = false,

    /// Start the guest on `model_dir` and wait until it serves. `ctx_size` 0
    /// leaves the context to the guest.
    pub fn start(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, ctx_size: u32) !*SushiGuest {
        const bin = try prepare(allocator, io);
        defer allocator.free(bin);

        var log_buf: [std.fs.max_path_bytes]u8 = undefined;
        const log_path = try logPath(&log_buf, homeDir(), options.host_port);
        const log_fd = try openLog(io, log_path);
        defer _ = std.c.close(log_fd);

        var secret: [32]u8 = undefined;
        io.random(&secret);
        const key = std.fmt.bytesToHex(secret, .lower);
        var env = try guestEnv(allocator, &key);
        defer env.deinit();

        const port = try pickPort(options.host_port);
        var port_buf: [8]u8 = undefined;
        var pid_buf: [16]u8 = undefined;
        var ctx_buf: [16]u8 = undefined;
        var argv_buf: [18][]const u8 = undefined;
        const argv = launchArgv(
            &argv_buf,
            bin,
            model_dir,
            std.fmt.bufPrint(&port_buf, "{d}", .{port}) catch unreachable,
            std.fmt.bufPrint(&pid_buf, "{d}", .{std.c.getpid()}) catch unreachable,
            if (ctx_size > 0) std.fmt.bufPrint(&ctx_buf, "{d}", .{ctx_size}) catch unreachable else null,
        );
        const self = try allocator.create(SushiGuest);
        const out: std.Io.File = .{ .handle = log_fd, .flags = .{ .nonblocking = false } };
        const child = std.process.spawn(io, .{ .argv = argv, .environ_map = &env, .stdin = .ignore, .stdout = .{ .file = out }, .stderr = .{ .file = out } }) catch |err| {
            allocator.destroy(self);
            log.err("[sushi] cannot start {s}: {s}\n", .{ bin, @errorName(err) });
            return error.SushiGuestUnrunnable;
        };
        self.* = .{ .allocator = allocator, .io = io, .pid = child.id.?, .port = port, .key = key };
        log.info("[sushi] guest pid={d} port={d} loading {s} (log: {s})\n", .{ self.pid, port, model_dir, log_path });
        errdefer self.stop();

        try self.waitReady();
        const models = try self.fetch(allocator, "/v1/models", fetch_timeout_ms);
        defer allocator.free(models);
        self.row = try parseRow(allocator, models);
        log.info("[sushi] guest ready: context {d}, kv {s}, mtp {}, quantization \"{s}\"\n", .{ self.row.context_length, self.row.kv_quant, self.row.mtp_loaded, self.row.quantization });
        return self;
    }

    /// SIGTERM, then SIGKILL after a grace period; reaps and frees the guest.
    pub fn stop(self: *SushiGuest) void {
        if (!self.hasExited()) {
            self.signal(.TERM);
            const deadline = io_util.nowMsMonotonic(self.io) + stop_grace_ms;
            while (!self.hasExited() and io_util.nowMsMonotonic(self.io) < deadline) {
                std.Io.sleep(self.io, .fromMilliseconds(50), .real) catch {};
            }
            if (!self.hasExited()) {
                log.warn("[sushi] guest pid={d} ignored SIGTERM; killing it\n", .{self.pid});
                self.signal(.KILL);
                self.reap(0);
            }
        }
        log.info("[sushi] guest pid={d} stopped\n", .{self.pid});
        self.row.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Whether the guest process is gone (it is reaped here if so).
    pub fn hasExited(self: *SushiGuest) bool {
        self.reap(std.c.W.NOHANG);
        return self.reaped;
    }

    /// Forward one request and pump the guest's response to `conn` unchanged.
    /// A guest that hangs up without a byte is unreachable too: nothing was sent.
    pub fn forward(self: *SushiGuest, method: []const u8, raw_path: []const u8, body: []const u8, conn: anytype) error{PeerUnreachable}!void {
        var auth_buf: [96]u8 = undefined;
        const auth = std.fmt.bufPrint(&auth_buf, "Authorization: Bearer {s}\r\n", .{&self.key}) catch unreachable;
        var counted: Counted(@TypeOf(conn)) = .{ .conn = conn };
        try lan.tunnelWithHeaders(self.remote(), method, raw_path, auth, body, &counted);
        if (counted.bytes == 0) return error.PeerUnreachable;
    }

    /// The body of the guest's 200 answer to `GET path` within `timeout_ms`. Caller frees.
    pub fn fetch(self: *SushiGuest, allocator: std.mem.Allocator, path: []const u8, timeout_ms: i64) ![]u8 {
        var capture: Capture = .{ .allocator = allocator, .io = self.io, .deadline_ms = io_util.nowMsMonotonic(self.io) + timeout_ms };
        defer capture.bytes.deinit(allocator);
        self.forward("GET", path, "", &capture) catch return error.SushiGuestUnreachable;
        if (capture.timed_out) return error.SushiGuestUnreachable;
        const body = okBody(capture.bytes.items) orelse return error.SushiGuestBadResponse;
        return allocator.dupe(u8, body);
    }

    fn remote(self: *const SushiGuest) lan.Remote {
        return .{ .ip4 = .{ 127, 0, 0, 1 }, .port = self.port };
    }

    fn reap(self: *SushiGuest, flags: c_int) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.reaped) return;
        var status: c_int = 0;
        const r = std.c.waitpid(self.pid, &status, flags);
        if (r == self.pid or r < 0) self.reaped = true;
    }

    fn signal(self: *SushiGuest, sig: std.c.SIG) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.reaped) _ = std.c.kill(self.pid, sig);
    }

    /// The guest binds its listener only once its model is loaded.
    fn waitReady(self: *SushiGuest) !void {
        const deadline = io_util.nowMsMonotonic(self.io) + ready_timeout_ms;
        while (true) {
            if (stopping.load(.acquire)) return error.SushiGuestCancelled;
            if (self.hasExited()) return error.SushiGuestExited;
            if (self.fetch(self.allocator, "/health", 2000)) |body| {
                self.allocator.free(body);
                return;
            } else |_| {}
            if (io_util.nowMsMonotonic(self.io) > deadline) return error.SushiGuestTimeout;
            std.Io.sleep(self.io, .fromMilliseconds(500), .real) catch {};
        }
    }
};

/// This process's environment plus the guest's key.
fn guestEnv(allocator: std.mem.Allocator, key: []const u8) !std.process.Environ.Map {
    var n: usize = 0;
    while (std.c.environ[n] != null) n += 1;
    const current: std.process.Environ = .{ .block = .{ .slice = std.c.environ[0..n :null] } };
    var env = try current.createMap(allocator);
    errdefer env.deinit();
    try env.put(key_env, key);
    return env;
}

/// A load failure's name for the registry. A guest that died or never came up
/// names its log: the reason is written there and nowhere else.
pub fn errorLabel(buf: []u8, err: anyerror) []const u8 {
    switch (err) {
        error.SushiGuestExited, error.SushiGuestTimeout => {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = logPath(&path_buf, homeDir(), options.host_port) catch return @errorName(err);
            return std.fmt.bufPrint(buf, "{s} (log: {s})", .{ @errorName(err), path }) catch @errorName(err);
        },
        else => return @errorName(err),
    }
}

pub fn logPath(buf: []u8, home: []const u8, host_port: u16) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/logs/sushi-{d}.log", .{ home, host_port });
}

/// The guest's `/props` with `settings.engine` naming it: to a client, the guest
/// is the engine serving this model. Everything else is the guest's own.
pub fn overlayProps(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{ .parse_numbers = false });
    defer parsed.deinit();
    if (parsed.value != .object) return error.SushiGuestBadResponse;
    if (parsed.value.object.getPtr("settings")) |settings| {
        if (settings.* == .object) try settings.object.put(parsed.arena.allocator(), "engine", .{ .string = engine_name });
    }
    return std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse "/tmp");
}

fn releaseUrl(allocator: std.mem.Allocator, tag: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "https://github.com/beamivalice/sushi/releases/download/{s}/" ++ release_asset, .{tag});
}

/// The pinned release's `sushi`, downloaded into
/// `~/.mlx-serve/engines/sushi/<tag>/` on first use. Caller frees.
fn ensureRelease(allocator: std.mem.Allocator, io: std.Io, home: []const u8) ![]u8 {
    const dir = try std.fmt.allocPrint(allocator, "{s}/.mlx-serve/engines/sushi/{s}", .{ home, release_tag });
    defer allocator.free(dir);
    const url = try releaseUrl(allocator, release_tag);
    defer allocator.free(url);
    return stageRelease(allocator, io, url, release_sha256, release_tree_sha256, dir);
}

/// `<dir>/sushi-macos-arm64/sushi`, its tree checked against `tree_sha256` on
/// every call. A tree that fails the check is replaced from the release tarball
/// at `url`, itself checked against `sha256` before extraction. Caller frees.
fn stageRelease(allocator: std.mem.Allocator, io: std.Io, url: []const u8, sha256: []const u8, tree_sha256: []const u8, dir: []const u8) ![]u8 {
    const tree = try std.fmt.allocPrint(allocator, "{s}/" ++ release_dir, .{dir});
    defer allocator.free(tree);
    const bin = try std.fmt.allocPrint(allocator, "{s}/sushi", .{tree});
    errdefer allocator.free(bin);
    if (treeSha256Hex(allocator, io, tree)) |got| {
        if (std.mem.eql(u8, &got, tree_sha256)) return bin;
        log.warn("[sushi] {s}: tree sha256 {s}, pinned {s}; re-staging\n", .{ tree, &got, tree_sha256 });
    } else |_| {}
    std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const staging = try std.fmt.allocPrint(allocator, "{s}.partial", .{dir});
    defer allocator.free(staging);
    const tarball = try std.fmt.allocPrint(allocator, "{s}/" ++ release_asset, .{staging});
    defer allocator.free(tarball);

    std.Io.Dir.cwd().deleteTree(io, staging) catch {};
    defer std.Io.Dir.cwd().deleteTree(io, staging) catch {};
    std.Io.Dir.cwd().createDirPath(io, staging) catch return error.SushiDownloadFailed;
    log.info("[sushi] downloading {s}\n", .{url});
    // A stalled transfer fails instead of holding the load forever.
    try runTool(io, &.{ "/usr/bin/curl", "-fsSL", "--retry", "3", "--connect-timeout", "30", "--speed-limit", "10000", "--speed-time", "60", "-o", tarball, url });
    const got = try sha256Hex(io, tarball);
    if (!std.mem.eql(u8, &got, sha256)) {
        log.err("[sushi] {s}: sha256 {s}, pinned {s}\n", .{ url, &got, sha256 });
        return error.SushiChecksumMismatch;
    }
    try runTool(io, &.{ "/usr/bin/tar", "-xzf", tarball, "-C", staging });
    std.Io.Dir.deleteFileAbsolute(io, tarball) catch {};
    const staged_tree = try std.fmt.allocPrint(allocator, "{s}/" ++ release_dir, .{staging});
    defer allocator.free(staged_tree);
    const got_tree = treeSha256Hex(allocator, io, staged_tree) catch return error.SushiDownloadFailed;
    if (!std.mem.eql(u8, &got_tree, tree_sha256)) {
        log.err("[sushi] {s}: tree sha256 {s}, pinned {s}\n", .{ url, &got_tree, tree_sha256 });
        return error.SushiChecksumMismatch;
    }
    std.Io.Dir.renameAbsolute(staging, dir, io) catch return error.SushiDownloadFailed;
    std.Io.Dir.accessAbsolute(io, bin, .{}) catch return error.SushiDownloadFailed;
    log.info("[sushi] staged {s}\n", .{dir});
    return bin;
}

/// Run a download step to completion, or kill it when the server stops.
fn runTool(io: std.Io, argv: []const []const u8) !void {
    const child = std.process.spawn(io, .{ .argv = argv, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore }) catch return error.SushiDownloadFailed;
    const pid = child.id.?;
    var status: c_int = 0;
    while (std.c.waitpid(pid, &status, std.c.W.NOHANG) == 0) {
        if (stopping.load(.acquire)) {
            _ = std.c.kill(pid, .KILL);
            _ = std.c.waitpid(pid, &status, 0);
            return error.SushiDownloadFailed;
        }
        std.Io.sleep(io, .fromMilliseconds(100), .real) catch {};
    }
    if (std.c.W.IFEXITED(@bitCast(status)) and std.c.W.EXITSTATUS(@bitCast(status)) == 0) return;
    log.err("[sushi] {s} failed (status {d})\n", .{ argv[0], status });
    return error.SushiDownloadFailed;
}

fn sha256Hex(io: std.Io, path: []const u8) ![64]u8 {
    var file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return error.SushiDownloadFailed;
    defer file.close(io);
    return fileSha256Hex(file);
}

/// sha256 over the sorted lines `<relative path>\x00<sha256 hex>\n` of every
/// regular file under `root` (fetch-sushi.sh computes the same). Anything added,
/// removed or changed moves it.
fn treeSha256Hex(allocator: std.mem.Allocator, io: std.Io, root: []const u8) ![64]u8 {
    var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch return error.SushiDownloadFailed;
    defer dir.close(io);
    const File = struct { path: []const u8, sha: [64]u8 };
    var files: std.ArrayList(File) = .empty;
    defer {
        for (files.items) |f| allocator.free(f.path);
        files.deinit(allocator);
    }
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    while (walker.next(io) catch return error.SushiDownloadFailed) |e| {
        if (e.kind != .file) continue;
        var file = e.dir.openFile(io, e.basename, .{}) catch return error.SushiDownloadFailed;
        defer file.close(io);
        const sha = try fileSha256Hex(file);
        const path = try allocator.dupe(u8, e.path);
        errdefer allocator.free(path);
        try files.append(allocator, .{ .path = path, .sha = sha });
    }
    std.mem.sort(File, files.items, {}, struct {
        fn lt(_: void, x: File, y: File) bool {
            return std.mem.lessThan(u8, x.path, y.path);
        }
    }.lt);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (files.items) |f| {
        hash.update(f.path);
        hash.update("\x00");
        hash.update(&f.sha);
        hash.update("\n");
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

fn fileSha256Hex(file: std.Io.File) ![64]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = std.c.read(file.handle, &buf, buf.len);
        if (n < 0) return error.SushiDownloadFailed;
        if (n == 0) break;
        hash.update(buf[0..@intCast(n)]);
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

/// Running the binary also proves it launches here (libraries found, not quarantined).
fn verifyGuest(allocator: std.mem.Allocator, io: std.Io, bin: []const u8) !void {
    const res = std.process.run(allocator, io, .{ .argv = &.{ bin, "--guest-manifest" }, .stdout_limit = .limited(64 << 10), .stderr_limit = .limited(64 << 10) }) catch |err| {
        log.err("[sushi] cannot run {s}: {s}\n", .{ bin, @errorName(err) });
        return error.SushiGuestUnrunnable;
    };
    defer allocator.free(res.stdout);
    defer allocator.free(res.stderr);
    switch (res.term) {
        .exited => |code| if (code != 0) return error.SushiGuestUnrunnable,
        else => return error.SushiGuestUnrunnable,
    }
    try checkManifest(allocator, res.stdout);
}

fn checkManifest(allocator: std.mem.Allocator, manifest: []const u8) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, manifest, .{}) catch return error.SushiManifestUnreadable;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SushiManifestUnreadable;
    const api = parsed.value.object.get("guest_api") orelse return error.SushiGuestApiUnsupported;
    if (api != .integer or api.integer != guest_api) return error.SushiGuestApiUnsupported;
}

/// One pack served on loopback. The guest logs to stderr only (the host hands
/// it the log file) and exits when this process does.
fn launchArgv(buf: *[18][]const u8, bin: []const u8, model_dir: []const u8, port: []const u8, parent_pid: []const u8, ctx_size: ?[]const u8) []const []const u8 {
    const fixed = [_][]const u8{ bin, "--serve", "--model", model_dir, "--host", "127.0.0.1", "--port", port, "--parent-pid", parent_pid, "--log-file", "off", "--api-key-env", key_env, "--api-key-strict" };
    @memcpy(buf[0..fixed.len], &fixed);
    const ctx = ctx_size orelse return buf[0..fixed.len];
    buf[fixed.len] = "--ctx-size";
    buf[fixed.len + 1] = ctx;
    return buf[0 .. fixed.len + 2];
}

/// The first free loopback port above the host's own.
fn pickPort(host_port: u16) !u16 {
    var port: u32 = @as(u32, host_port) + 1;
    while (port <= @min(@as(u32, host_port) + 64, 65535)) : (port += 1) {
        if (portFree(@intCast(port))) return @intCast(port);
    }
    return error.SushiNoFreePort;
}

fn portFree(port: u16) bool {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    return std.c.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) == 0;
}

fn openLog(io: std.Io, path: []const u8) !std.c.fd_t {
    if (std.fs.path.dirname(path)) |parent| std.Io.Dir.cwd().createDirPath(io, parent) catch {};
    var z: [std.fs.max_path_bytes + 3]u8 = undefined;
    const path_z = std.mem.printSentinel(&z, "{s}", .{path}, 0) catch return error.NameTooLong;
    const flags: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .CLOEXEC = true };
    var fd = std.c.open(path_z, flags, @as(std.c.mode_t, 0o644));
    if (fd >= 0 and std.c.lseek(fd, 0, std.c.SEEK.END) > log_max_bytes) {
        _ = std.c.close(fd);
        var old: [std.fs.max_path_bytes + 3]u8 = undefined;
        _ = std.c.rename(path_z, std.mem.printSentinel(&old, "{s}.1", .{path}, 0) catch return error.NameTooLong);
        fd = std.c.open(path_z, flags, @as(std.c.mode_t, 0o644));
    }
    if (fd < 0) return error.SushiLogUnwritable;
    return fd;
}

/// A duck-typed `conn` for `lan.tunnel` that keeps the response in memory and
/// hangs up at its deadline.
const Capture = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    deadline_ms: i64,
    bytes: std.ArrayList(u8) = .empty,
    timed_out: bool = false,

    pub fn writeAll(self: *Capture, data: []const u8) !void {
        if (self.bytes.items.len + data.len > 16 << 20) return error.ResponseTooLarge;
        try self.bytes.appendSlice(self.allocator, data);
    }
    pub fn peerClosed(self: *Capture) bool {
        self.timed_out = io_util.nowMsMonotonic(self.io) > self.deadline_ms;
        return self.timed_out;
    }
};

/// A `conn` wrapper that counts what reached it.
fn Counted(comptime Conn: type) type {
    return struct {
        conn: Conn,
        bytes: usize = 0,

        pub fn writeAll(self: *@This(), data: []const u8) !void {
            try self.conn.writeAll(data);
            self.bytes += data.len;
        }
        pub fn peerClosed(self: *@This()) bool {
            return self.conn.peerClosed();
        }
    };
}

/// The body of a raw HTTP response, or null unless its status is 200.
fn okBody(raw: []const u8) ?[]const u8 {
    const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return null;
    if (std.mem.indexOf(u8, raw[0..line_end], " 200 ") == null) return null;
    const head_end = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return null;
    return raw[head_end + 4 ..];
}

/// What the guest reports for the model it serves; `/v1/models` shows these
/// instead of values mlx-serve would compute for its own engine.
pub const Row = struct {
    context_length: u32 = 0,
    /// "" when unknown; `jsonSafe` text only (spliced into JSON unescaped).
    quantization: []const u8 = "",
    kv_quant: []const u8 = "",
    mtp_loaded: bool = false,

    fn deinit(self: *Row, allocator: std.mem.Allocator) void {
        allocator.free(self.quantization);
        allocator.free(self.kv_quant);
        self.* = .{};
    }
};

/// The first `/v1/models` row: the guest's one model. Caller deinits.
fn parseRow(allocator: std.mem.Allocator, body: []const u8) !Row {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.SushiGuestBadResponse;
    defer parsed.deinit();
    if (parsed.value != .object) return error.SushiGuestBadResponse;
    const data = parsed.value.object.get("data") orelse return error.SushiGuestBadResponse;
    if (data != .array or data.array.items.len == 0 or data.array.items[0] != .object) return error.SushiGuestBadResponse;
    const first = data.array.items[0].object;
    const meta: ?std.json.ObjectMap = if (first.get("meta")) |m| (if (m == .object) m.object else null) else null;
    const ctx = first.get("context_length");
    const mtp = if (meta) |m| m.get("mtp_loaded") else null;
    return .{
        .context_length = if (ctx != null and ctx.? == .integer) std.math.cast(u32, ctx.?.integer) orelse 0 else 0,
        .quantization = try jsonSafe(allocator, if (meta) |m| m.get("quantization") else null),
        .kv_quant = try jsonSafe(allocator, if (meta) |m| m.get("kv_quant") else null),
        .mtp_loaded = mtp != null and mtp.? == .bool and mtp.?.bool,
    };
}

/// A dupe of a short printable-ASCII string with no `"` or `\\`; "" otherwise.
fn jsonSafe(allocator: std.mem.Allocator, v: ?std.json.Value) ![]const u8 {
    const s = if (v) |x| (if (x == .string) x.string else "") else "";
    if (s.len > 64) return allocator.dupe(u8, "");
    for (s) |c| if (c < 0x20 or c > 0x7e or c == '"' or c == '\\') return allocator.dupe(u8, "");
    return allocator.dupe(u8, s);
}

const testing = std.testing;

test "sushi: the pin names a release tag, its asset URL and a sha256" {
    try testing.expect(std.mem.startsWith(u8, release_tag, "v"));
    for ([_][]const u8{ release_sha256, release_tree_sha256 }) |pin| {
        try testing.expectEqual(@as(usize, 64), pin.len);
        for (pin) |c| try testing.expect(std.ascii.isDigit(c) or (c >= 'a' and c <= 'f'));
    }
    const url = try releaseUrl(testing.allocator, "v1.0.0");
    defer testing.allocator.free(url);
    try testing.expectEqualStrings("https://github.com/beamivalice/sushi/releases/download/v1.0.0/sushi-bin-macos-arm64.tar.gz", url);
}

test "sushi: a guest_api this host does not speak is refused by name" {
    try checkManifest(testing.allocator, "{\"version\":\"1.0.0\",\"guest_api\":1,\"model_types\":[\"qwen4_exp\"]}");
    try testing.expectError(error.SushiGuestApiUnsupported, checkManifest(testing.allocator, "{\"version\":\"2.0.0\",\"guest_api\":2}"));
    try testing.expectError(error.SushiGuestApiUnsupported, checkManifest(testing.allocator, "{\"version\":\"0.9.0\"}"));
    try testing.expectError(error.SushiGuestApiUnsupported, checkManifest(testing.allocator, "{\"guest_api\":\"1\"}"));
    try testing.expectError(error.SushiManifestUnreadable, checkManifest(testing.allocator, "sushi 1.0.0"));
}

test "sushi: the guest serves one pack on loopback and is tied to this process" {
    var buf: [18][]const u8 = undefined;
    const base = [_][]const u8{ "/e/sushi", "--serve", "--model", "/m/pack", "--host", "127.0.0.1", "--port", "18601", "--parent-pid", "42", "--log-file", "off", "--api-key-env", key_env, "--api-key-strict" };
    const plain = launchArgv(&buf, "/e/sushi", "/m/pack", "18601", "42", null);
    try testing.expectEqual(base.len, plain.len);
    for (base, plain) |want, got| try testing.expectEqualStrings(want, got);
    const sized = launchArgv(&buf, "/e/sushi", "/m/pack", "18601", "42", "65536");
    try testing.expectEqualStrings("--ctx-size", sized[base.len]);
    try testing.expectEqualStrings("65536", sized[base.len + 1]);
}

test "sushi: a guest failure names its log" {
    var buf: [std.fs.max_path_bytes + 64]u8 = undefined;
    const label = errorLabel(&buf, error.SushiGuestExited);
    try testing.expect(std.mem.startsWith(u8, label, "SushiGuestExited (log: "));
    try testing.expect(std.mem.endsWith(u8, label, "/.mlx-serve/logs/sushi-11234.log)"));
    try testing.expectEqualStrings("SushiChecksumMismatch", errorLabel(&buf, error.SushiChecksumMismatch));
}

test "sushi: /props names the engine and keeps the guest's fields" {
    const props = try overlayProps(testing.allocator,
        \\{"memory":{"active_bytes":52613432320,"cache_bytes":0},"settings":{"engine":"mlx","kv_quant":"8","mtp":{"acceptance_param":0.95}}}
    );
    defer testing.allocator.free(props);
    try testing.expectEqualStrings(
        \\{"memory":{"active_bytes":52613432320,"cache_bytes":0},"settings":{"engine":"sushi","kv_quant":"8","mtp":{"acceptance_param":0.95}}}
    , props);
    try testing.expectError(error.SushiGuestBadResponse, overlayProps(testing.allocator, "[]"));
}

test "sushi: the guest's answers are read from 200 responses only" {
    try testing.expectEqualStrings("{\"status\":\"ok\"}", okBody("HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}").?);
    try testing.expect(okBody("HTTP/1.1 503 Service Unavailable\r\n\r\n{}") == null);
    try testing.expect(okBody("HTTP/1.1 200 OK\r\n") == null);
}

test "sushi: the guest's own row describes the model it serves" {
    const a = testing.allocator;
    var row = try parseRow(a,
        \\{"object":"list","data":[{"id":"pack","context_length":262144,"meta":{"quantization":"EXL3 3bpw experts, 8-bit dense","kv_quant":"8","mtp_loaded":true}},{"id":"other","context_length":8}]}
    );
    defer row.deinit(a);
    try testing.expectEqual(@as(u32, 262144), row.context_length);
    try testing.expectEqualStrings("EXL3 3bpw experts, 8-bit dense", row.quantization);
    try testing.expectEqualStrings("8", row.kv_quant);
    try testing.expect(row.mtp_loaded);
    // A value our JSON could not carry unescaped is dropped, not spliced in.
    var odd = try parseRow(a,
        \\{"data":[{"context_length":4096,"meta":{"quantization":"8\"-bit","kv_quant":"a\\b"}}]}
    );
    defer odd.deinit(a);
    try testing.expectEqual(@as(u32, 4096), odd.context_length);
    try testing.expectEqualStrings("", odd.quantization);
    try testing.expectEqualStrings("", odd.kv_quant);
    try testing.expect(!odd.mtp_loaded);
    try testing.expectError(error.SushiGuestBadResponse, parseRow(a, "{\"object\":\"list\",\"data\":[]}"));
}

test "sushi: the guest takes the first free port above the host's" {
    const taken = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    try testing.expect(taken >= 0);
    defer _ = std.c.close(taken);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    try testing.expectEqual(@as(c_int, 0), std.c.bind(taken, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)));
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    try testing.expectEqual(@as(c_int, 0), std.c.getsockname(taken, @ptrCast(&sa), &len));
    const taken_port = std.mem.bigToNative(u16, sa.port);
    const port = try pickPort(taken_port - 1);
    try testing.expect(port > taken_port);
}

test "sushi: stop reaps a running guest, and a guest that died reads as exited" {
    const io = testing.io;
    const sleeper = try std.process.spawn(io, .{ .argv = &.{ "/bin/sleep", "30" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    const pid = sleeper.id.?;
    const guest = try testing.allocator.create(SushiGuest);
    guest.* = .{ .allocator = testing.allocator, .io = io, .pid = pid, .port = 0, .key = @splat('k') };
    try testing.expect(!guest.hasExited());
    guest.stop();
    try testing.expectEqual(@as(c_int, -1), std.c.kill(pid, @enumFromInt(0)));

    const quitter = try std.process.spawn(io, .{ .argv = &.{"/usr/bin/true"}, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    var dead: SushiGuest = .{ .allocator = testing.allocator, .io = io, .pid = quitter.id.?, .port = 0, .key = @splat('k') };
    const deadline = io_util.nowMsMonotonic(io) + 5000;
    while (!dead.hasExited() and io_util.nowMsMonotonic(io) < deadline) std.Io.sleep(io, .fromMilliseconds(10), .real) catch {};
    try testing.expect(dead.hasExited());
}

/// A release tarball in `root` with a `sushi`, a dylib and a metallib; its tarball
/// sha256 and its tree digest.
const TestRelease = struct { url: []u8, sha: [64]u8, tree_sha: [64]u8 };

fn testRelease(a: std.mem.Allocator, io: std.Io, tmp: *std.testing.TmpDir, root: []const u8) !TestRelease {
    try tmp.dir.createDirPath(io, "src/" ++ release_dir ++ "/lib");
    try tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ release_dir ++ "/sushi", .data = "#!/bin/sh\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ release_dir ++ "/lib/libmlx.dylib", .data = "mlx" });
    try tmp.dir.writeFile(io, .{ .sub_path = "src/" ++ release_dir ++ "/lib/mlx.metallib", .data = "kernels" });
    const src = try std.fs.path.join(a, &.{ root, "src" });
    defer a.free(src);
    const tarball = try std.fs.path.join(a, &.{ root, release_asset });
    defer a.free(tarball);
    try runTool(io, &.{ "/usr/bin/tar", "-czf", tarball, "-C", src, release_dir });
    const tree = try std.fs.path.join(a, &.{ src, release_dir });
    defer a.free(tree);
    return .{ .url = try std.fmt.allocPrint(a, "file://{s}", .{tarball}), .sha = try sha256Hex(io, tarball), .tree_sha = try treeSha256Hex(a, io, tree) };
}

test "sushi: the tree digest is sha256 over sorted `path NUL sha256 LF` lines of every file" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "t/lib");
    try tmp.dir.writeFile(io, .{ .sub_path = "t/sushi", .data = "abc" });
    try tmp.dir.writeFile(io, .{ .sub_path = "t/lib/libmlx.dylib", .data = "" });
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tree = try std.fs.path.join(a, &.{ root_buf[0..try tmp.dir.realPath(io, &root_buf)], "t" });
    defer a.free(tree);
    // `lib/…` sorts before `sushi`; the second hash is sha256("abc").
    var want = std.crypto.hash.sha2.Sha256.init(.{});
    want.update("lib/libmlx.dylib\x00e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n");
    want.update("sushi\x00ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n");
    try testing.expectEqualStrings(&std.fmt.bytesToHex(want.finalResult(), .lower), &(try treeSha256Hex(a, io, tree)));
}

test "sushi: a release is staged only when its tarball and its tree match the pin" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    const rel = try testRelease(a, io, &tmp, root);
    defer a.free(rel.url);
    const zeros = "0000000000000000000000000000000000000000000000000000000000000000";

    const refused = try std.fs.path.join(a, &.{ root, "engines/refused" });
    defer a.free(refused);
    try testing.expectError(error.SushiChecksumMismatch, stageRelease(a, io, rel.url, zeros, &rel.tree_sha, refused));
    try testing.expectError(error.SushiChecksumMismatch, stageRelease(a, io, rel.url, &rel.sha, zeros, refused));
    try testing.expectError(error.FileNotFound, std.Io.Dir.accessAbsolute(io, refused, .{}));

    const dir = try std.fs.path.join(a, &.{ root, "engines/v1.0.0" });
    defer a.free(dir);
    // A tree left without its binary is stale, not in the way.
    try tmp.dir.createDirPath(io, "engines/v1.0.0/" ++ release_dir);
    const bin = try stageRelease(a, io, rel.url, &rel.sha, &rel.tree_sha, dir);
    defer a.free(bin);
    try std.Io.Dir.accessAbsolute(io, bin, .{});
    // Staged and unchanged: a later start never fetches again.
    const again = try stageRelease(a, io, "file:///nonexistent", &rel.sha, &rel.tree_sha, dir);
    defer a.free(again);
    try testing.expectEqualStrings(bin, again);
}

test "sushi: a tampered, added or removed file in the staged tree is re-staged, never loaded" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    const rel = try testRelease(a, io, &tmp, root);
    defer a.free(rel.url);
    const dir = try std.fs.path.join(a, &.{ root, "engines/v1.0.0" });
    defer a.free(dir);
    const tree = try std.fs.path.join(a, &.{ dir, release_dir });
    defer a.free(tree);
    a.free(try stageRelease(a, io, rel.url, &rel.sha, &rel.tree_sha, dir));

    const staged = "engines/v1.0.0/" ++ release_dir;
    const edits = [_]struct { path: []const u8, data: ?[]const u8 }{
        .{ .path = staged ++ "/sushi", .data = "#!/bin/sh\ntouch ran\n" },
        .{ .path = staged ++ "/lib/libmlx.dylib", .data = "evil" },
        .{ .path = staged ++ "/lib/mlx.metallib", .data = "evil" },
        .{ .path = staged ++ "/lib/libinject.dylib", .data = "evil" },
        .{ .path = staged ++ "/lib/mlx.metallib", .data = null },
    };
    for (edits) |e| {
        if (e.data) |d| try tmp.dir.writeFile(io, .{ .sub_path = e.path, .data = d }) else try tmp.dir.deleteFile(io, e.path);
        a.free(try stageRelease(a, io, rel.url, &rel.sha, &rel.tree_sha, dir));
        try testing.expectEqualStrings(&rel.tree_sha, &(try treeSha256Hex(a, io, tree)));
    }

    // Offline, the tampered tree is removed and the start fails by name.
    try tmp.dir.writeFile(io, .{ .sub_path = staged ++ "/lib/libmlx.dylib", .data = "evil" });
    try testing.expectError(error.SushiDownloadFailed, stageRelease(a, io, "file:///nonexistent", &rel.sha, &rel.tree_sha, dir));
    try testing.expectError(error.FileNotFound, std.Io.Dir.accessAbsolute(io, dir, .{}));
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "ran", .{}));
}

/// A loopback listener for fake guests; its port lands in `port`.
fn testListener(port: *u16) !std.c.fd_t {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.Sock;
    errdefer _ = std.c.close(fd);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (std.c.bind(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.Sock;
    var len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    if (std.c.getsockname(fd, @ptrCast(&sa), &len) != 0 or std.c.listen(fd, 4) != 0) return error.Sock;
    port.* = std.mem.bigToNative(u16, sa.port);
    return fd;
}

test "sushi: every request to the guest carries its key, and a silent guest is unreachable" {
    const io = testing.io;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer _ = std.c.close(lst);
    const Peer = struct {
        fn run(l: std.c.fd_t) void {
            // First connection: answer only when it carries the key.
            const c = std.c.accept(l, null, null);
            var req: [2048]u8 = undefined;
            const n = std.c.read(c, &req, req.len);
            const keyed = n > 0 and std.mem.indexOf(u8, req[0..@intCast(n)], "Authorization: Bearer " ++ @as([64]u8, @splat('k')) ++ "\r\n") != null;
            const reply = if (keyed) "HTTP/1.1 200 OK\r\n\r\n{\"ok\":1}" else "HTTP/1.1 401 Unauthorized\r\n\r\n";
            _ = std.c.write(c, reply.ptr, reply.len);
            _ = std.c.close(c);
            // Second: accept and hang up without a byte.
            _ = std.c.close(std.c.accept(l, null, null));
        }
    };
    const th = try std.Thread.spawn(.{}, Peer.run, .{lst});
    defer th.join();
    var guest: SushiGuest = .{ .allocator = testing.allocator, .io = io, .pid = 0, .port = port, .key = @splat('k') };
    const body = try guest.fetch(testing.allocator, "/props", 2000);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("{\"ok\":1}", body);
    var capture: Capture = .{ .allocator = testing.allocator, .io = io, .deadline_ms = std.math.maxInt(i64) };
    defer capture.bytes.deinit(testing.allocator);
    try testing.expectError(error.PeerUnreachable, guest.forward("POST", "/v1/chat/completions", "{}", &capture));
}

test "sushi: a guest that accepts and never answers is given up on" {
    const io = testing.io;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer _ = std.c.close(lst);
    var guest: SushiGuest = .{ .allocator = testing.allocator, .io = io, .pid = 0, .port = port, .key = @splat('k') };
    try testing.expectError(error.SushiGuestUnreachable, guest.fetch(testing.allocator, "/health", 100));
}

test "sushi: a shutdown stops the wait for the guest and a running download" {
    const io = testing.io;
    stopping.store(true, .release);
    defer stopping.store(false, .release);
    const start_ms = io_util.nowMsMonotonic(io);
    try testing.expectError(error.SushiDownloadFailed, runTool(io, &.{ "/bin/sleep", "30" }));
    const sleeper = try std.process.spawn(io, .{ .argv = &.{ "/bin/sleep", "30" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore });
    const guest = try testing.allocator.create(SushiGuest);
    guest.* = .{ .allocator = testing.allocator, .io = io, .pid = sleeper.id.?, .port = 1, .key = @splat('k') };
    try testing.expectError(error.SushiGuestCancelled, guest.waitReady());
    guest.stop();
    try testing.expect(io_util.nowMsMonotonic(io) - start_ms < 5000);
}

test "sushi: an engine that cannot start is refused before anything is fetched" {
    options.disabled = true;
    defer options.disabled = false;
    try testing.expectError(error.SushiDisabled, prepare(testing.allocator, testing.io));
}

test "sushi: other children never inherit the guest log" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)];
    const path = try std.fs.path.join(testing.allocator, &.{ root, "logs/sushi-1.log" });
    defer testing.allocator.free(path);
    const fd = try openLog(testing.io, path);
    defer _ = std.c.close(fd);
    try testing.expect(std.c.fcntl(fd, std.c.F.GETFD) & std.c.FD_CLOEXEC != 0);
}

test "sushi: only failures that say nothing about the pack are retried" {
    try testing.expect(isTransient(error.SushiDownloadFailed));
    try testing.expect(isTransient(error.SushiGuestTimeout));
    try testing.expect(!isTransient(error.SushiGuestExited));
    try testing.expect(!isTransient(error.SushiChecksumMismatch));
    try testing.expectEqual(@as(?TransientError, error.SushiNoFreePort), transientFromName("SushiNoFreePort"));
    try testing.expectEqual(@as(?TransientError, null), transientFromName("MissingWeight"));
}
