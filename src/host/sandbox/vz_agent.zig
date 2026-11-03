//! vz-agent — the guest half of the Agent Sandbox.
//!
//! A tiny static `aarch64-linux-musl` ELF injected into the rootfs before boot
//! (exactly like `/.vz-init`), so it works with ANY base image — the bundled
//! one for the Mac App Store build, or a user-chosen `python:3.12-slim` on the
//! Developer ID build. Zero dependencies beyond libc.
//!
//! It replaces the console-based transports:
//!
//!   before: hvc1 = one persistent `/bin/sh` driven by `ShellSentinel` (a tty,
//!           so echo-proofing, ANSI collapse and CRLF handling were all load-
//!           bearing); hvc2 = an `=EOS=`-framed shell monitor loop.
//!   after:  vsock. One connection per process, real exit codes, stdout and
//!           stderr separated, no tty, no sentinel nonce. hvc0 (kernel printk)
//!           is the only console left.
//!
//! Because `VZVirtioSocketDevice.connect(toPort:)` opens as many connections as
//! the host wants, MCP servers get a connection each and no multiplexing
//! protocol is needed.
//!
//! ## Wire format
//!
//! Every message on a connection is `[u8 channel][u32 length big-endian][payload]`.
//!
//! The request payload is a length-prefixed binary record, NOT JSON. Both ends
//! are ours, and hand-rolled JSON escaping is a bug class this codebase has
//! already paid for twice (see the tool-call escaping gotchas). Binary framing
//! has no escaping to get wrong, and the golden bytes are shared with the Swift
//! `GuestProtocol` tests.
//!
//! Zig 0.16 gutted `std.posix` (only `poll`/`read` survive; the rest moved
//! behind `std.Io`), so — like `log.zig` — this file talks to libc directly.

const std = @import("std");
const builtin = @import("builtin");

// ─── libc ────────────────────────────────────────────────────────────────────

const c = struct {
    extern "c" fn fork() c_int;
    extern "c" fn execve(path: [*:0]const u8, argv: [*:null]const ?[*:0]const u8, envp: [*:null]const ?[*:0]const u8) c_int;
    extern "c" fn pipe(fds: *[2]c_int) c_int;
    extern "c" fn dup2(old: c_int, new: c_int) c_int;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
    extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
    extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
    extern "c" fn kill(pid: c_int, sig: c_int) c_int;
    extern "c" fn setsid() c_int;
    extern "c" fn chdir(path: [*:0]const u8) c_int;
    // `open` is VARIADIC in C. Declaring `mode` as a fixed third parameter is
    // an ABI mismatch on Darwin arm64 (variadic args go on the stack, named
    // ones in registers), so the kernel read a GARBAGE creation mode — 0o400
    // in Debug by luck, 0o000 in ReleaseFast, where the detached-log test
    // caught it. Linux aarch64 happens to pass both the same way, which is
    // exactly why only the macOS test build saw it.
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    // Loose `usize` signature so `SIG_IGN` (the integer 1) needs no fake
    // function pointer — mirrors `log.zig`'s libc style.
    extern "c" fn signal(sig: c_int, handler: usize) usize;
    extern "c" fn usleep(usec: c_uint) c_int;
    extern "c" fn _exit(code: c_int) noreturn;
    extern "c" fn socket(domain: c_int, sock_type: c_int, protocol: c_int) c_int;
    extern "c" fn bind(fd: c_int, addr: *const anyopaque, len: u32) c_int;
    extern "c" fn listen(fd: c_int, backlog: c_int) c_int;
    extern "c" fn accept(fd: c_int, addr: ?*anyopaque, len: ?*u32) c_int;
    extern "c" fn poll(fds: [*]PollFd, n: c_ulong, timeout: c_int) c_int;
};

pub const PollFd = extern struct {
    fd: c_int,
    events: i16,
    revents: i16,
};

const POLLIN: i16 = 0x001;
const POLLHUP: i16 = 0x010;
const POLLERR: i16 = 0x008;

// `open` flags differ between Linux and Darwin, and this file is compiled for
// BOTH (Linux = the guest, macOS = `zig build test`). Getting them wrong fails
// silently: `open` returns -1, the `dup2` is skipped, and a detached child's
// output escapes to the agent's own stdout instead of its log file.
const O_RDONLY: c_int = 0;
const O_WRONLY: c_int = if (builtin.os.tag == .linux) 0o1 else 0x0001;
const O_CREAT: c_int = if (builtin.os.tag == .linux) 0o100 else 0x0200;
const O_APPEND: c_int = if (builtin.os.tag == .linux) 0o2000 else 0x0008;

const SIGTERM: c_int = 15;
const SIGKILL: c_int = 9;
const SIGPIPE: c_int = 13;
const SIG_IGN: usize = 1;
const WNOHANG: c_int = 1;

// ─── Ports ───────────────────────────────────────────────────────────────────

/// One connection == one process. Used for the agent's `shell` tool AND for
/// every stdio MCP server.
///
/// The hvc2 guest-monitor loop (memory / IP / listening ports) deliberately
/// stays where it is for now: it works, it is independent of the shell
/// channel, and moving it would widen this change's blast radius for no gain.
/// Port 1025 is reserved for it.
pub const port_exec: u32 = 1024;

// ─── Frame codec (pure — mirrored by Swift `GuestProtocol`) ──────────────────

pub const Channel = enum(u8) {
    /// host → guest: the `Request` record. Always the first frame.
    request = 0,
    /// host → guest: bytes for the child's stdin.
    stdin = 1,
    /// host → guest: close the child's stdin (payload empty).
    stdin_eof = 2,
    /// guest → host: bytes the child wrote to stdout.
    stdout = 3,
    /// guest → host: bytes the child wrote to stderr.
    stderr = 4,
    /// guest → host: the child's exit status, as a 4-byte big-endian i32.
    exit = 5,
    /// guest → host: the child's pid, as a 4-byte big-endian i32. Sent once,
    /// before any output, so a detached process can later be killed.
    started = 6,
    /// guest → host: the agent itself failed (UTF-8 message). Terminal.
    err = 7,

    pub fn fromByte(b: u8) ?Channel {
        return switch (b) {
            0...7 => @enumFromInt(b),
            else => null,
        };
    }
};

pub const header_len = 5;
/// A single frame's payload ceiling. Guards against a corrupt length turning
/// into a multi-gigabyte allocation.
pub const max_payload = 8 << 20;

pub fn encodeHeader(channel: Channel, len: u32) [header_len]u8 {
    var out: [header_len]u8 = undefined;
    out[0] = @intFromEnum(channel);
    std.mem.writeInt(u32, out[1..5], len, .big);
    return out;
}

pub const HeaderError = error{ BadChannel, PayloadTooLarge };

pub fn decodeHeader(bytes: *const [header_len]u8) HeaderError!struct { channel: Channel, len: u32 } {
    const channel = Channel.fromByte(bytes[0]) orelse return error.BadChannel;
    const len = std.mem.readInt(u32, bytes[1..5], .big);
    if (len > max_payload) return error.PayloadTooLarge;
    return .{ .channel = channel, .len = len };
}

// ─── Request record (pure — mirrored by Swift `GuestProtocol`) ───────────────

/// Binary layout, all lengths u32 big-endian:
///
///   [u8 flags]                       bit0 = detach
///   [u32 len][bytes]                 command   (run as `/bin/sh -c <command>`)
///   [u32 len][bytes]                 cwd       (empty = don't chdir)
///   [u32 len][bytes]                 log_path  (detach only; empty = /dev/null)
///   [u32 count]                      env pairs
///     [u32 len][bytes][u32 len][bytes]  key, value
pub const Request = struct {
    command: []const u8,
    cwd: []const u8 = "",
    log_path: []const u8 = "",
    detach: bool = false,
    env: []const EnvPair = &.{},

    pub const EnvPair = struct { key: []const u8, value: []const u8 };

    pub fn encode(self: Request, allocator: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        try out.append(allocator, if (self.detach) 1 else 0);
        for ([_][]const u8{ self.command, self.cwd, self.log_path }) |field| {
            try appendLengthPrefixed(&out, allocator, field);
        }
        var count: [4]u8 = undefined;
        std.mem.writeInt(u32, &count, @intCast(self.env.len), .big);
        try out.appendSlice(allocator, &count);
        for (self.env) |pair| {
            try appendLengthPrefixed(&out, allocator, pair.key);
            try appendLengthPrefixed(&out, allocator, pair.value);
        }
        return out.toOwnedSlice(allocator);
    }

    pub const DecodeError = error{ Truncated, OutOfMemory };

    /// The returned slices point INTO `bytes`; `env` is owned by the caller.
    pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!Request {
        var cursor: usize = 0;
        if (bytes.len < 1) return error.Truncated;
        const flags = bytes[0];
        cursor = 1;

        const command = try readLengthPrefixed(bytes, &cursor);
        const cwd = try readLengthPrefixed(bytes, &cursor);
        const log_path = try readLengthPrefixed(bytes, &cursor);

        if (cursor + 4 > bytes.len) return error.Truncated;
        const count = std.mem.readInt(u32, bytes[cursor..][0..4], .big);
        cursor += 4;
        // A bogus count must not preallocate gigabytes: every pair costs at
        // least 8 bytes of header, so the remaining bytes bound it.
        if (count > (bytes.len - cursor) / 8 + 1) return error.Truncated;

        const env = try allocator.alloc(EnvPair, count);
        errdefer allocator.free(env);
        for (env) |*pair| {
            pair.key = try readLengthPrefixed(bytes, &cursor);
            pair.value = try readLengthPrefixed(bytes, &cursor);
        }

        return .{
            .command = command,
            .cwd = cwd,
            .log_path = log_path,
            .detach = flags & 1 != 0,
            .env = env,
        };
    }
};

fn appendLengthPrefixed(out: *std.ArrayList(u8), allocator: std.mem.Allocator, bytes: []const u8) !void {
    var len: [4]u8 = undefined;
    std.mem.writeInt(u32, &len, @intCast(bytes.len), .big);
    try out.appendSlice(allocator, &len);
    try out.appendSlice(allocator, bytes);
}

fn readLengthPrefixed(bytes: []const u8, cursor: *usize) error{Truncated}![]const u8 {
    if (cursor.* + 4 > bytes.len) return error.Truncated;
    const len = std.mem.readInt(u32, bytes[cursor.*..][0..4], .big);
    cursor.* += 4;
    if (cursor.* + len > bytes.len) return error.Truncated;
    defer cursor.* += len;
    return bytes[cursor.*..][0..len];
}

// ─── Exit status ─────────────────────────────────────────────────────────────

/// Decode `waitpid`'s status word the way a shell reports it: a signalled
/// process becomes 128 + signal, matching `$?`.
pub fn exitCodeFromStatus(status: c_int) i32 {
    const s: u32 = @bitCast(status);
    if (s & 0x7f == 0) return @intCast((s >> 8) & 0xff); // exited normally
    const signal: i32 = @intCast(s & 0x7f);
    return 128 + signal;
}

// ─── I/O helpers ─────────────────────────────────────────────────────────────

extern "c" var environ: [*:null]?[*:0]u8;

const IoError = error{ Closed, WriteFailed };

fn writeAll(fd: c_int, bytes: []const u8) IoError!void {
    var sent: usize = 0;
    while (sent < bytes.len) {
        const n = c.write(fd, bytes.ptr + sent, bytes.len - sent);
        if (n <= 0) return error.WriteFailed;
        sent += @intCast(n);
    }
}

pub fn writeFrame(fd: c_int, channel: Channel, payload: []const u8) IoError!void {
    const header = encodeHeader(channel, @intCast(payload.len));
    try writeAll(fd, &header);
    if (payload.len > 0) try writeAll(fd, payload);
}

/// Reads exactly `buf.len` bytes, or reports the peer hung up.
fn readExact(fd: c_int, buf: []u8) IoError!void {
    var got: usize = 0;
    while (got < buf.len) {
        const n = c.read(fd, buf.ptr + got, buf.len - got);
        if (n <= 0) return error.Closed;
        got += @intCast(n);
    }
}

fn writeExit(fd: c_int, code: i32) void {
    var payload: [4]u8 = undefined;
    std.mem.writeInt(i32, &payload, code, .big);
    writeFrame(fd, .exit, &payload) catch {};
}

fn writeStarted(fd: c_int, pid: i32) void {
    var payload: [4]u8 = undefined;
    std.mem.writeInt(i32, &payload, pid, .big);
    writeFrame(fd, .started, &payload) catch {};
}

// ─── Child process ───────────────────────────────────────────────────────────

/// `execve`-ready argv/envp. Built in the PARENT: between `fork` and `execve`
/// only async-signal-safe calls are legal, and allocating is not one of them.
const Spawn = struct {
    argv: [:null]?[*:0]const u8,
    envp: [:null]?[*:0]const u8,

    fn build(gpa: std.mem.Allocator, req: Request) !Spawn {
        const command = try gpa.dupeSentinel(u8, req.command, 0);
        var argv = try gpa.allocSentinel(?[*:0]const u8, 3, null);
        argv[0] = "sh";
        argv[1] = "-c";
        argv[2] = command.ptr;

        var envp: std.ArrayList(?[*:0]const u8) = .empty;
        // Inherit the agent's environment (which `/.vz-init` seeded from the
        // OCI image config), minus anything the request overrides.
        var i: usize = 0;
        while (environ[i]) |entry| : (i += 1) {
            const text = std.mem.span(entry);
            const eq = std.mem.indexOfScalar(u8, text, '=') orelse continue;
            const overridden = for (req.env) |pair| {
                if (std.mem.eql(u8, pair.key, text[0..eq])) break true;
            } else false;
            if (!overridden) try envp.append(gpa, entry);
        }
        for (req.env) |pair| {
            const entry = try std.fmt.allocPrintSentinel(gpa, "{s}={s}", .{ pair.key, pair.value }, 0);
            try envp.append(gpa, entry.ptr);
        }
        try envp.append(gpa, null);
        const owned = try envp.toOwnedSlice(gpa);

        return .{
            .argv = argv,
            .envp = @ptrCast(owned[0 .. owned.len - 1 :null]),
        };
    }
};

/// Runs in the forked child. Never returns.
fn childExec(spawn: Spawn, cwd_z: ?[*:0]const u8) noreturn {
    // Drop every inherited descriptor above stdio before exec. The fork copied
    // the agent's whole fd table — the vsock listener, THIS connection, and the
    // pipe originals. A backgrounded daemon holding the connection socket keeps
    // it half-open after the agent closes it, and nothing the child runs should
    // be able to write frames to the host. (Callers dup2 their stdio first.)
    var fd: c_int = 3;
    while (fd < 1024) : (fd += 1) _ = c.close(fd);
    if (cwd_z) |dir| _ = c.chdir(dir);
    _ = c.execve("/bin/sh", spawn.argv.ptr, spawn.envp.ptr);
    c._exit(127); // exec failed — the shell's own "command not found" code
}

/// SIGTERM, a real grace interval, then SIGKILL any survivor. The waitpid here
/// uses WNOHANG only to PROBE — the caller still does the reaping wait.
fn killChild(pid: c_int) void {
    _ = c.kill(pid, SIGTERM);
    var status: c_int = 0;
    var waited_us: c_uint = 0;
    while (waited_us < 500_000) : (waited_us += 20_000) {
        if (c.waitpid(pid, &status, WNOHANG) != 0) return; // exited (or gone)
        _ = c.usleep(20_000);
    }
    _ = c.kill(pid, SIGKILL);
}

// ─── Serving one connection ──────────────────────────────────────────────────

/// Handle exactly one request on `sock`: spawn the process, stream its output
/// back, report its exit status. Closing `sock` from the host kills the child —
/// that is how a timeout or a cancelled tool call is expressed.
///
/// OS-agnostic on purpose: the tests drive it over a `socketpair` on macOS, so
/// everything except the vsock bind/accept is covered without a guest.
pub fn serveConnection(gpa: std.mem.Allocator, sock: c_int) void {
    // One arena per connection: `Spawn.build` and the request decode allocate
    // freely (argv/envp duplications) and nothing frees them individually — a
    // long-lived agent serving thousands of shell commands must not leak a few
    // KB per command.
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    serveConnectionInner(arena.allocator(), sock) catch |err| {
        var buf: [128]u8 = undefined;
        const msg = std.fmt.bufPrint(&buf, "vz-agent: {s}", .{@errorName(err)}) catch "vz-agent: error";
        writeFrame(sock, .err, msg) catch {};
    };
}

fn serveConnectionInner(gpa: std.mem.Allocator, sock: c_int) !void {
    // 1. The request frame.
    var header: [header_len]u8 = undefined;
    try readExact(sock, &header);
    const head = try decodeHeader(&header);
    if (head.channel != .request) return error.ExpectedRequest;

    const payload = try gpa.alloc(u8, head.len);
    defer gpa.free(payload);
    try readExact(sock, payload);

    const req = try Request.decode(gpa, payload);
    defer gpa.free(req.env);

    const spawn = try Spawn.build(gpa, req);
    const cwd_z: ?[*:0]const u8 = if (req.cwd.len > 0) (try gpa.dupeSentinel(u8, req.cwd, 0)).ptr else null;

    if (req.detach) return detachedRun(gpa, sock, spawn, cwd_z, req.log_path);
    return foregroundRun(gpa, sock, spawn, cwd_z);
}

fn foregroundRun(gpa: std.mem.Allocator, sock: c_int, spawn: Spawn, cwd_z: ?[*:0]const u8) !void {
    _ = gpa;
    var in_fds: [2]c_int = undefined;
    var out_fds: [2]c_int = undefined;
    var err_fds: [2]c_int = undefined;
    if (c.pipe(&in_fds) != 0 or c.pipe(&out_fds) != 0 or c.pipe(&err_fds) != 0) return error.PipeFailed;

    const pid = c.fork();
    if (pid < 0) return error.ForkFailed;
    if (pid == 0) {
        _ = c.dup2(in_fds[0], 0);
        _ = c.dup2(out_fds[1], 1);
        _ = c.dup2(err_fds[1], 2);
        _ = c.close(in_fds[1]);
        _ = c.close(out_fds[0]);
        _ = c.close(err_fds[0]);
        childExec(spawn, cwd_z);
    }

    _ = c.close(in_fds[0]);
    _ = c.close(out_fds[1]);
    _ = c.close(err_fds[1]);
    defer _ = c.close(in_fds[1]);

    writeStarted(sock, pid);
    pump(sock, in_fds[1], out_fds[0], err_fds[0], pid);

    var status: c_int = 0;
    _ = c.waitpid(pid, &status, 0);
    writeExit(sock, exitCodeFromStatus(status));
}

/// Double-fork so the surviving process reparents to init and never becomes a
/// zombie the agent has to reap. The middle child reports the grandchild's pid
/// back through a pipe; the host uses it to `kill` the process later.
fn detachedRun(gpa: std.mem.Allocator, sock: c_int, spawn: Spawn, cwd_z: ?[*:0]const u8, log_path: []const u8) !void {
    const log_z: [*:0]const u8 = if (log_path.len > 0)
        (try gpa.dupeSentinel(u8, log_path, 0)).ptr
    else
        "/dev/null";

    var pid_fds: [2]c_int = undefined;
    if (c.pipe(&pid_fds) != 0) return error.PipeFailed;

    const middle = c.fork();
    if (middle < 0) return error.ForkFailed;
    if (middle == 0) {
        _ = c.close(pid_fds[0]);
        _ = c.setsid();
        const grandchild = c.fork();
        if (grandchild == 0) {
            const log = c.open(log_z, O_WRONLY | O_CREAT | O_APPEND, @as(c_uint, 0o644));
            const devnull = c.open("/dev/null", O_RDONLY);
            if (devnull >= 0) _ = c.dup2(devnull, 0);
            if (log >= 0) {
                _ = c.dup2(log, 1);
                _ = c.dup2(log, 2);
            }
            childExec(spawn, cwd_z);
        }
        var buf: [4]u8 = undefined;
        std.mem.writeInt(i32, &buf, grandchild, .big);
        _ = c.write(pid_fds[1], &buf, 4);
        c._exit(0);
    }

    _ = c.close(pid_fds[1]);
    var buf: [4]u8 = undefined;
    readExact(pid_fds[0], &buf) catch {};
    _ = c.close(pid_fds[0]);

    var status: c_int = 0;
    _ = c.waitpid(middle, &status, 0); // reap the middle child immediately

    writeStarted(sock, std.mem.readInt(i32, &buf, .big));
    writeExit(sock, 0); // the process is running; the host has its pid
}

/// Shuttle bytes until the child's stdout AND stderr both hit EOF. A hangup on
/// `sock` means the host gave up (timeout, cancelled tool call) — kill the child.
fn pump(sock: c_int, child_stdin: c_int, child_stdout: c_int, child_stderr: c_int, pid: c_int) void {
    defer _ = c.close(child_stdout);
    defer _ = c.close(child_stderr);

    var fds = [_]PollFd{
        .{ .fd = sock, .events = POLLIN, .revents = 0 },
        .{ .fd = child_stdout, .events = POLLIN, .revents = 0 },
        .{ .fd = child_stderr, .events = POLLIN, .revents = 0 },
    };
    var open_streams: u2 = 2;
    var buf: [64 * 1024]u8 = undefined;

    while (open_streams > 0) {
        if (c.poll(&fds, fds.len, -1) < 0) break;

        // Host → child stdin, or the host hung up.
        if (fds[0].revents & (POLLIN | POLLHUP | POLLERR) != 0) {
            if (!forwardHostFrame(sock, child_stdin)) {
                killChild(pid);
                return;
            }
        }

        for ([_]usize{ 1, 2 }) |i| {
            if (fds[i].revents & (POLLIN | POLLHUP) == 0) continue;
            const n = c.read(fds[i].fd, &buf, buf.len);
            if (n <= 0) {
                fds[i].fd = -1; // poll ignores negative fds
                open_streams -= 1;
                continue;
            }
            const channel: Channel = if (i == 1) .stdout else .stderr;
            writeFrame(sock, channel, buf[0..@intCast(n)]) catch {
                // The host hung up mid-write. Same as a hangup seen by poll:
                // kill the child rather than leaking it to run forever.
                killChild(pid);
                return;
            };
        }
    }
}

/// Reads one frame from the host. Returns false when the connection is done
/// (hangup, or a frame we can't honor) — the caller then kills the child.
fn forwardHostFrame(sock: c_int, child_stdin: c_int) bool {
    var header: [header_len]u8 = undefined;
    readExact(sock, &header) catch return false;
    const head = decodeHeader(&header) catch return false;

    switch (head.channel) {
        .stdin => {
            var remaining = head.len;
            var buf: [32 * 1024]u8 = undefined;
            while (remaining > 0) {
                const want = @min(remaining, buf.len);
                readExact(sock, buf[0..want]) catch return false;
                writeAll(child_stdin, buf[0..want]) catch return false;
                remaining -= want;
            }
            return true;
        },
        .stdin_eof => {
            _ = c.close(child_stdin);
            return true;
        },
        else => return false,
    }
}

// ─── vsock listener (Linux guest only) ───────────────────────────────────────

const AF_VSOCK: c_int = 40;
const SOCK_STREAM: c_int = 1;
const VMADDR_CID_ANY: u32 = 0xFFFF_FFFF;

const sockaddr_vm = extern struct {
    family: u16 = @intCast(AF_VSOCK),
    reserved1: u16 = 0,
    port: u32,
    cid: u32,
    flags: u8 = 0,
    zero: [3]u8 = .{ 0, 0, 0 },
};

/// A unix-domain listener, used ONLY by the cross-language interop test. It
/// lets the Swift `GuestExec` driver talk to the REAL Zig agent on macOS, so
/// the two implementations are proven against each other long before a guest
/// kernel with vsock exists. Same `serveConnection`, different accept path.
const AF_UNIX: c_int = 1;

const sockaddr_un = extern struct {
    /// Darwin has `sun_len` as the first byte; Linux treats it as part of the
    /// 16-bit family. Setting it is correct on Darwin and harmless on Linux
    /// only because we never bind a unix socket there.
    len: u8 = @sizeOf(sockaddr_un),
    family: u8 = @intCast(AF_UNIX),
    path: [104]u8 = @splat(0),
};

extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]const u8;

/// Set by the interop test to a unix-socket path. Absent inside the guest.
pub const unix_socket_env = "VZ_AGENT_UNIX_SOCKET";

pub fn main() !void {
    // A host that hangs up while we're mid-write must surface as EPIPE from
    // `write` (handled), not SIGPIPE — whose default disposition would kill
    // the WHOLE agent, taking every other connection (all MCP servers + the
    // shell) down with it. Same reason the Swift side sets SO_NOSIGPIPE.
    _ = c.signal(SIGPIPE, SIG_IGN);

    // We link libc, and connections are served on their own threads.
    const gpa = std.heap.c_allocator;

    const listener = blk: {
        // Interop-test mode: `VZ_AGENT_UNIX_SOCKET=/path/to/sock vz-agent`.
        if (getenv(unix_socket_env)) |raw| {
            const path = std.mem.span(raw);
            var addr = sockaddr_un{};
            if (path.len + 1 > addr.path.len) return error.PathTooLong;
            @memcpy(addr.path[0..path.len], path);

            const fd = c.socket(AF_UNIX, SOCK_STREAM, 0);
            if (fd < 0) return error.SocketFailed;
            _ = unlink(@ptrCast(&addr.path)); // a stale socket file blocks bind
            if (c.bind(fd, &addr, @sizeOf(sockaddr_un)) != 0) return error.BindFailed;
            break :blk fd;
        }

        // Inside the guest: vsock.
        if (builtin.os.tag != .linux) {
            std.debug.print("vz-agent runs inside the Linux guest (or set {s} for the interop test)\n", .{unix_socket_env});
            return error.NoTransport;
        }
        const fd = c.socket(AF_VSOCK, SOCK_STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        var addr = sockaddr_vm{ .port = port_exec, .cid = VMADDR_CID_ANY };
        if (c.bind(fd, &addr, @sizeOf(sockaddr_vm)) != 0) return error.BindFailed;
        break :blk fd;
    };

    if (c.listen(listener, 16) != 0) return error.ListenFailed;

    while (true) {
        const conn = c.accept(listener, null, null);
        if (conn < 0) continue;
        // One thread per connection: a long-lived MCP server must not block the
        // agent's shell, and vice versa.
        const thread = std.Thread.spawn(.{}, connectionThread, .{ gpa, conn }) catch {
            _ = c.close(conn);
            continue;
        };
        thread.detach();
    }
}

fn connectionThread(gpa: std.mem.Allocator, conn: c_int) void {
    defer _ = c.close(conn);
    serveConnection(gpa, conn);
}

// ─── Tests (pure; run on macOS via `zig build test`) ─────────────────────────

test "frame header round-trips" {
    const encoded = encodeHeader(.stdout, 12345);
    const decoded = try decodeHeader(&encoded);
    try std.testing.expectEqual(Channel.stdout, decoded.channel);
    try std.testing.expectEqual(@as(u32, 12345), decoded.len);
}

test "frame header is exactly the bytes Swift expects" {
    // GOLDEN — GuestProtocolTests.swift asserts the same five bytes. Changing
    // either side without the other silently desyncs the transport.
    try std.testing.expectEqualSlices(u8, &.{ 3, 0x00, 0x00, 0x30, 0x39 }, &encodeHeader(.stdout, 12345));
    try std.testing.expectEqualSlices(u8, &.{ 0, 0x00, 0x00, 0x00, 0x00 }, &encodeHeader(.request, 0));
    try std.testing.expectEqualSlices(u8, &.{ 5, 0x00, 0x00, 0x00, 0x04 }, &encodeHeader(.exit, 4));
}

test "decodeHeader rejects an unknown channel and an absurd length" {
    try std.testing.expectError(error.BadChannel, decodeHeader(&.{ 9, 0, 0, 0, 1 }));
    try std.testing.expectError(error.PayloadTooLarge, decodeHeader(&.{ 3, 0xff, 0xff, 0xff, 0xff }));
}

test "request encoding is exactly the bytes Swift produces" {
    // GOLDEN — `GuestProtocolTests.testRequestEncodingGoldenBytes` builds the
    // same buffer by hand. Round-tripping our own encoder proves nothing about
    // the OTHER language's; only fixed bytes on both sides do.
    const allocator = std.testing.allocator;
    const request = Request{
        .command = "ls",
        .cwd = "/w",
        .detach = true,
        .env = &.{.{ .key = "A", .value = "B" }},
    };
    const encoded = try request.encode(allocator);
    defer allocator.free(encoded);

    try std.testing.expectEqualSlices(u8, &.{
        1, // flags: detach
        0, 0, 0, 2, 'l', 's', // command
        0, 0, 0, 2, '/', 'w', // cwd
        0, 0, 0, 0, //          log_path (empty)
        0, 0, 0, 1, //          env count
        0, 0, 0, 1, 'A', //     key
        0, 0, 0, 1, 'B', //     value
    }, encoded);

    // And the decoder accepts those exact bytes.
    const decoded = try Request.decode(allocator, encoded);
    defer allocator.free(decoded.env);
    try std.testing.expectEqualStrings("ls", decoded.command);
    try std.testing.expectEqualStrings("/w", decoded.cwd);
    try std.testing.expect(decoded.detach);
    try std.testing.expectEqualStrings("B", decoded.env[0].value);
}

test "request round-trips through encode/decode" {
    const allocator = std.testing.allocator;
    const original = Request{
        .command = "echo hi && printf 'x\\ty'",
        .cwd = "/workspace/sub dir",
        .log_path = "/tmp/bg.log",
        .detach = true,
        .env = &.{
            .{ .key = "PATH", .value = "/usr/bin:/bin" },
            .{ .key = "WEIRD", .value = "a\"b\\c\nd" }, // no escaping to get wrong
        },
    };

    const bytes = try original.encode(allocator);
    defer allocator.free(bytes);
    const decoded = try Request.decode(allocator, bytes);
    defer allocator.free(decoded.env);

    try std.testing.expectEqualStrings(original.command, decoded.command);
    try std.testing.expectEqualStrings(original.cwd, decoded.cwd);
    try std.testing.expectEqualStrings(original.log_path, decoded.log_path);
    try std.testing.expect(decoded.detach);
    try std.testing.expectEqual(@as(usize, 2), decoded.env.len);
    try std.testing.expectEqualStrings("PATH", decoded.env[0].key);
    try std.testing.expectEqualStrings("a\"b\\c\nd", decoded.env[1].value);
}

test "request decode rejects truncation instead of reading past the buffer" {
    const allocator = std.testing.allocator;
    const original = Request{ .command = "ls", .env = &.{.{ .key = "A", .value = "B" }} };
    const bytes = try original.encode(allocator);
    defer allocator.free(bytes);

    var i: usize = 0;
    while (i < bytes.len) : (i += 1) {
        const result = Request.decode(allocator, bytes[0..i]);
        if (result) |ok| {
            allocator.free(ok.env);
            try std.testing.expect(false); // every prefix is truncated
        } else |err| try std.testing.expectEqual(error.Truncated, err);
    }
}

test "request decode rejects a bogus env count without allocating gigabytes" {
    const allocator = std.testing.allocator;
    // flags + three empty fields + count = 0xFFFFFFFF
    var bytes: [1 + 12 + 4]u8 = @splat(0);
    @memset(bytes[13..17], 0xff);
    try std.testing.expectError(error.Truncated, Request.decode(allocator, &bytes));
}

test "exit status maps signals the way a shell does" {
    try std.testing.expectEqual(@as(i32, 0), exitCodeFromStatus(0));
    try std.testing.expectEqual(@as(i32, 7), exitCodeFromStatus(7 << 8));
    try std.testing.expectEqual(@as(i32, 128 + 9), exitCodeFromStatus(9)); // SIGKILL
    try std.testing.expectEqual(@as(i32, 128 + 15), exitCodeFromStatus(15)); // SIGTERM
}

// ─── End-to-end over a socketpair ────────────────────────────────────────────
//
// `serveConnection` is OS-agnostic, so the whole request → spawn → stream →
// exit path is exercised here on macOS. Only the vsock bind/accept in `main`
// needs a real guest, and that is the part with nothing to get wrong.

extern "c" fn socketpair(domain: c_int, sock_type: c_int, protocol: c_int, fds: *[2]c_int) c_int;
const usleep = c.usleep;
extern "c" fn getpid() c_int;
// `AF_UNIX` and `unlink` are declared above, for the interop-test listener.

const Harness = struct {
    host: c_int,
    thread: std.Thread,

    fn start(req: Request) !Harness {
        var fds: [2]c_int = undefined;
        try std.testing.expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0);

        const thread = try std.Thread.spawn(.{}, serveThread, .{fds[1]});
        const self = Harness{ .host = fds[0], .thread = thread };

        const payload = try req.encode(std.testing.allocator);
        defer std.testing.allocator.free(payload);
        try writeFrame(self.host, .request, payload);
        return self;
    }

    fn serveThread(fd: c_int) void {
        // The serving side owns its end of the pair.
        defer _ = c.close(fd);
        serveConnection(std.heap.c_allocator, fd);
    }

    fn deinit(self: *Harness) void {
        _ = c.close(self.host);
        self.thread.join();
    }

    /// Collect frames until `.exit` (or `.err`), concatenating stdout/stderr.
    const Collected = struct {
        stdout: std.ArrayList(u8) = .empty,
        stderr: std.ArrayList(u8) = .empty,
        exit_code: ?i32 = null,
        pid: ?i32 = null,
        err_message: ?[]u8 = null,

        fn deinit(self: *Collected, gpa: std.mem.Allocator) void {
            self.stdout.deinit(gpa);
            self.stderr.deinit(gpa);
            if (self.err_message) |m| gpa.free(m);
        }
    };

    fn collect(self: *Harness, gpa: std.mem.Allocator) !Collected {
        var out = Collected{};
        errdefer out.deinit(gpa);
        while (true) {
            var header: [header_len]u8 = undefined;
            readExact(self.host, &header) catch break;
            const head = try decodeHeader(&header);
            const payload = try gpa.alloc(u8, head.len);
            defer gpa.free(payload);
            try readExact(self.host, payload);

            switch (head.channel) {
                .stdout => try out.stdout.appendSlice(gpa, payload),
                .stderr => try out.stderr.appendSlice(gpa, payload),
                .started => out.pid = std.mem.readInt(i32, payload[0..4], .big),
                .exit => {
                    out.exit_code = std.mem.readInt(i32, payload[0..4], .big);
                    break;
                },
                .err => {
                    out.err_message = try gpa.dupe(u8, payload);
                    break;
                },
                else => return error.UnexpectedChannel,
            }
        }
        return out;
    }
};

test "e2e: stdout, exit code, and a real pid" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{ .command = "echo hello; exit 3" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);

    try std.testing.expectEqualStrings("hello\n", got.stdout.items);
    try std.testing.expectEqual(@as(i32, 3), got.exit_code.?);
    try std.testing.expect(got.pid.? > 0);
}

test "e2e: stdout and stderr stay separated" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{ .command = "echo out; echo err 1>&2" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);

    try std.testing.expectEqualStrings("out\n", got.stdout.items);
    try std.testing.expectEqualStrings("err\n", got.stderr.items);
    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);
}

test "e2e: a signalled child reports 128 + signal, not 0" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{ .command = "kill -TERM $$" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqual(@as(i32, 128 + 15), got.exit_code.?);
}

test "e2e: cwd is honored" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{ .command = "pwd", .cwd = "/tmp" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    // macOS /tmp is a symlink to /private/tmp; the shell prints what it chdir'd to.
    try std.testing.expect(std.mem.indexOf(u8, got.stdout.items, "tmp") != null);
    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);
}

test "e2e: request env overrides the inherited environment" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{
        .command = "printf '%s' \"$VZ_TEST_VAR\"",
        .env = &.{.{ .key = "VZ_TEST_VAR", .value = "injected" }},
    });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqualStrings("injected", got.stdout.items);
}

test "e2e: the inherited environment survives when not overridden" {
    const gpa = std.testing.allocator;
    // PATH comes from the agent's own environ; without it `sh -c` still runs,
    // but the child must be able to see it.
    var h = try Harness.start(.{ .command = "test -n \"$PATH\" && echo has-path" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqualStrings("has-path\n", got.stdout.items);
}

test "e2e: stdin frames reach the child, and stdin_eof closes it" {
    const gpa = std.testing.allocator;
    var h = try Harness.start(.{ .command = "cat" });
    defer h.deinit();

    try writeFrame(h.host, .stdin, "ping");
    try writeFrame(h.host, .stdin_eof, "");

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqualStrings("ping", got.stdout.items);
    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);
}

test "e2e: a large payload survives framing" {
    const gpa = std.testing.allocator;
    // 300 KB crosses both the 64 KB read buffer and any single-frame assumption.
    var h = try Harness.start(.{ .command = "yes abcdefghij | head -30000" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 30000 * 11), got.stdout.items.len);
    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);
}

test "e2e: host hangup kills a running child instead of leaking it" {
    var h = try Harness.start(.{ .command = "echo up; sleep 60" });

    // Wait for proof the child is alive, then hang up mid-run.
    var header: [header_len]u8 = undefined;
    try readExact(h.host, &header); // started
    const started = try decodeHeader(&header);
    try std.testing.expectEqual(Channel.started, started.channel);
    var pid_buf: [4]u8 = undefined;
    try readExact(h.host, &pid_buf);
    const pid = std.mem.readInt(i32, &pid_buf, .big);

    try readExact(h.host, &header); // stdout "up\n"
    var out_buf: [3]u8 = undefined;
    try readExact(h.host, &out_buf);

    h.deinit(); // closes the socket; `pump` must SIGTERM the child

    // The child is gone: reaped by the agent, so `kill(pid, 0)` fails.
    // (A leaked `sleep 60` would still answer.)
    _ = usleep(200_000);
    try std.testing.expect(c.kill(pid, 0) != 0);
}

test "e2e: a detached process outlives the connection and reports its pid" {
    const gpa = std.testing.allocator;
    // libc, not std.fs: this file deliberately carries no `std.Io`.
    const log_file = try std.fmt.allocPrintSentinel(gpa, "/tmp/vz-agent-detach-{d}.log", .{getpid()}, 0);
    defer gpa.free(log_file);
    defer _ = unlink(log_file.ptr);

    var h = try Harness.start(.{
        .command = "echo backgrounded",
        .detach = true,
        .log_path = log_file,
    });

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    h.deinit();

    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);
    try std.testing.expect(got.pid.? > 0);

    // The output landed in the log, not on the connection.
    try std.testing.expectEqualStrings("", got.stdout.items);
    _ = usleep(300_000);

    const fd = c.open(log_file.ptr, O_RDONLY);
    try std.testing.expect(fd >= 0);
    defer _ = c.close(fd);
    var contents: [64]u8 = undefined;
    const n = c.read(fd, &contents, contents.len);
    try std.testing.expect(n > 0);
    try std.testing.expectEqualStrings("backgrounded\n", contents[0..@intCast(n)]);

    // The log must be created 0644 — RED when `open` is declared with a fixed
    // `mode` parameter instead of variadic: Darwin arm64 passes variadic args
    // on the stack, so the kernel saw garbage (0o400 in Debug, 0o000 in
    // ReleaseFast, where the O_RDONLY reopen above failed outright). The Stat
    // layout below is Darwin's, so the check is macOS-only; Linux aarch64
    // passes variadic args like named ones and never saw the symptom.
    if (builtin.os.tag == .macos) {
        var st: DarwinStat = undefined;
        try std.testing.expect(darwinStat(log_file.ptr, &st) == 0);
        try std.testing.expectEqual(@as(u16, 0o644), st.mode & 0o777);
    }
}

/// Darwin arm64 `struct stat` — just enough to read `st_mode`. Test-only.
const DarwinStat = extern struct {
    dev: i32,
    mode: u16,
    nlink: u16,
    ino: u64,
    uid: u32,
    gid: u32,
    rdev: i32,
    atime: [2]i64,
    mtime: [2]i64,
    ctime: [2]i64,
    btime: [2]i64,
    size: i64,
    blocks: i64,
    blksize: i32,
    flags: u32,
    gen: u32,
    lspare: i32,
    qspare: [2]i64,
};
extern "c" fn stat(path: [*:0]const u8, buf: *DarwinStat) c_int;
const darwinStat = stat;

test "e2e: a child does not inherit the agent's descriptors" {
    const gpa = std.testing.allocator;
    // /dev/fd lists the CHILD's open descriptors. Without the close loop in
    // `childExec` the fork leaks the whole fd table — this connection's socket,
    // the pipe originals, every test-runner fd — into anything the guest runs,
    // and a backgrounded daemon holding the connection socket keeps it
    // half-open after the agent is done with it. Expect exactly stdio plus the
    // descriptor `ls` itself opens to read the directory.
    var h = try Harness.start(.{ .command = "ls /dev/fd" });
    defer h.deinit();

    var got = try h.collect(gpa);
    defer got.deinit(gpa);
    try std.testing.expectEqual(@as(i32, 0), got.exit_code.?);

    var it = std.mem.tokenizeScalar(u8, got.stdout.items, '\n');
    while (it.next()) |line| {
        const listed = try std.fmt.parseInt(i32, line, 10);
        try std.testing.expect(listed <= 4); // 0-2 stdio, 3-4 ls's own dirfd
    }
}

test "e2e: a malformed first frame yields an error frame, not a hang" {
    const gpa = std.testing.allocator;
    var fds: [2]c_int = undefined;
    try std.testing.expect(socketpair(AF_UNIX, SOCK_STREAM, 0, &fds) == 0);
    const thread = try std.Thread.spawn(.{}, Harness.serveThread, .{fds[1]});
    defer thread.join();
    defer _ = c.close(fds[0]);

    // `.stdout` where `.request` is required.
    try writeFrame(fds[0], .stdout, "nope");

    var header: [header_len]u8 = undefined;
    try readExact(fds[0], &header);
    const head = try decodeHeader(&header);
    try std.testing.expectEqual(Channel.err, head.channel);

    const payload = try gpa.alloc(u8, head.len);
    defer gpa.free(payload);
    try readExact(fds[0], payload);
    try std.testing.expect(std.mem.indexOf(u8, payload, "ExpectedRequest") != null);
}
