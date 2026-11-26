//! Zig 0.17's test runner (lib/compiler/test_runner.zig) with these changes. `MLX_SERVE_TEST_SHARD=k/n` keeps one
//! shard of the tests, plus every test named `sharded: …` (it runs in every process), so `zig build test` runs n copies
//! of one binary in parallel. With `--plan=<file>` (`build/test_plan.tsv`: each test's measured time and lane) the
//! shards are balanced by time, longest first, and `--lane=cpu|gpu|serial` keeps one lane's tests; without a plan a
//! shard is every n-th test. `--test-filter=<s>` (repeatable) keeps, at run time, the tests whose name contains any of
//! them and every unnamed test (as Zig's compile-time filter does), so one binary serves every filter. `--list` prints
//! this process's tests and runs none; `--skip=<n>` resumes after the first n (a gate restarting a crashed shard);
//! `--times-out=<file>` appends one record per test (status, wall and CPU time, MLX peak bytes: what
//! `scripts/test_plan.py` builds the plan from). Unset = every test, as std's runner. And it runs the test root's
//! process setup (`msv_test_process_init`) before the first test and `msv_test_enter` before each.
const builtin = @import("builtin");

const std = @import("std");
const Io = std.Io;
const fatal = std.process.fatal;
const testing = std.testing;
const assert = std.debug.assert;
const panic = std.debug.panic;
const fuzz_abi = std.Build.abi.fuzz;

pub const std_options: std.Options = .{
    .logFn = log,
};

var log_err_count: usize = 0;
var fba: std.heap.FixedBufferAllocator = .init(&fba_buffer);
var fba_buffer: [65536]u8 = undefined;
var stdin_buffer: [4096]u8 = undefined;
var stdout_buffer: [4096]u8 = undefined;
var stdin_reader: Io.File.Reader = undefined;
var stdout_writer: Io.File.Writer = undefined;
const runner_threaded_io: Io = Io.Threaded.global_single_threaded.io();

/// Keep in sync with logic in `std.Build.addRunArtifact` which decides whether
/// the test runner will communicate with the build runner via `std.zig.Server`.
const need_simple = switch (builtin.zig_backend) {
    .stage2_aarch64,
    .stage2_loongarch,
    .stage2_powerpc,
    .stage2_riscv64,
    => true,
    else => false,
};

/// This process's tests: `builtin.test_functions` narrowed by the options below.
var tests: []const std.builtin.TestFn = &.{};

const Lane = enum { cpu, gpu, serial };

/// The command line's selection options (the rest of the arguments are std's).
var opt_plan: ?[:0]const u8 = null;
var opt_lane: ?Lane = null;
var opt_filters: std.ArrayList([]const u8) = .empty;
var opt_list = false;
var opt_skip: usize = 0;
var opt_times_out: ?[:0]const u8 = null;

/// A test missing from the plan (measured under this many ms, or added since) is placed as a CPU test of this many ms.
const unplanned_ms: u64 = 20;

const Planned = struct { ms: u64, lane: Lane };

fn isSharded(t: std.builtin.TestFn) bool {
    return std.mem.indexOf(u8, t.name, ".sharded: ") != null;
}

/// An unnamed `test {}` (`<namespace>.test_<n>`): a filter keeps it, as Zig's compile-time `-Dtest-filter` does.
fn isUnnamed(t: std.builtin.TestFn) bool {
    if (std.mem.indexOf(u8, t.name, ".test.") != null) return false;
    const dot = std.mem.lastIndexOfScalar(u8, t.name, '.') orelse return false;
    return std.mem.startsWith(u8, t.name[dot + 1 ..], "test_");
}

fn readFile(path: [:0]const u8) []u8 {
    const gpa = std.heap.page_allocator;
    const fd = std.c.open(path, .{}, @as(c_uint, 0));
    if (fd < 0) panic("--plan: cannot open {s}", .{path});
    defer _ = std.c.close(fd);
    var out: std.ArrayList(u8) = .empty;
    while (true) {
        out.ensureUnusedCapacity(gpa, 1 << 16) catch panic("out of memory", .{});
        const room = out.unusedCapacitySlice();
        const n = std.c.read(fd, room.ptr, room.len);
        if (n < 0) panic("--plan: cannot read {s}", .{path});
        if (n == 0) break;
        out.items.len += @intCast(n);
    }
    return out.items;
}

/// `<lane>\t<ms>\t<test name>` per line; `#` starts a comment.
fn loadPlan(path: [:0]const u8) std.StringHashMapUnmanaged(Planned) {
    const gpa = std.heap.page_allocator;
    var map: std.StringHashMapUnmanaged(Planned) = .empty;
    var lines = std.mem.splitScalar(u8, readFile(path), '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        var f = std.mem.splitScalar(u8, line, '\t');
        const lane = std.meta.stringToEnum(Lane, f.next() orelse continue) orelse panic("--plan: bad lane in {s}", .{line});
        const ms = std.fmt.parseUnsigned(u64, f.next() orelse continue, 10) catch panic("--plan: bad ms in {s}", .{line});
        const name = f.rest();
        map.put(gpa, name, .{ .ms = ms, .lane = lane }) catch panic("out of memory", .{});
    }
    return map;
}

fn shardEnv() struct { k: usize, n: usize } {
    if (builtin.os.tag == .windows) return .{ .k = 0, .n = 1 };
    const raw = std.mem.span(std.c.getenv("MLX_SERVE_TEST_SHARD") orelse return .{ .k = 0, .n = 1 });
    const slash = std.mem.indexOfScalar(u8, raw, '/') orelse panic("MLX_SERVE_TEST_SHARD: want k/n, got {s}", .{raw});
    const k = std.fmt.parseUnsigned(usize, raw[0..slash], 10) catch panic("MLX_SERVE_TEST_SHARD: {s}", .{raw});
    const n = std.fmt.parseUnsigned(usize, raw[slash + 1 ..], 10) catch panic("MLX_SERVE_TEST_SHARD: {s}", .{raw});
    if (n == 0 or k >= n) panic("MLX_SERVE_TEST_SHARD: {s}", .{raw});
    return .{ .k = k, .n = n };
}

const ByTime = struct {
    ms: []const u64,
    fn longerFirst(c: ByTime, a: u32, b: u32) bool {
        return c.ms[a] > c.ms[b] or (c.ms[a] == c.ms[b] and a < b);
    }
};

/// Filters, then the lane, then this shard (by time with a plan, else every n-th), then `--skip`.
fn selectTests() void {
    const gpa = std.heap.page_allocator;
    const all = builtin.test_functions;
    var kept: std.ArrayList(std.builtin.TestFn) = .empty;
    for (all) |t| {
        if (opt_filters.items.len > 0 and !isUnnamed(t)) {
            for (opt_filters.items) |f| {
                if (std.mem.indexOf(u8, t.name, f) != null) break;
            } else continue;
        }
        kept.append(gpa, t) catch panic("out of memory", .{});
    }
    const sh = shardEnv();
    var plan: std.StringHashMapUnmanaged(Planned) = .empty;
    if (opt_plan) |p| plan = loadPlan(p);
    const planned = struct {
        fn of(m: *const std.StringHashMapUnmanaged(Planned), t: std.builtin.TestFn) Planned {
            return m.get(t.name) orelse .{ .ms = unplanned_ms, .lane = .cpu };
        }
    }.of;
    if (opt_lane) |lane| {
        if (opt_plan == null) panic("--lane needs --plan", .{});
        var len: usize = 0;
        for (kept.items) |t| if (isSharded(t) or planned(&plan, t).lane == lane) {
            kept.items[len] = t;
            len += 1;
        };
        kept.items.len = len;
    }
    const mine = gpa.alloc(bool, kept.items.len) catch panic("out of memory", .{});
    @memset(mine, false);
    if (opt_plan != null and sh.n > 1) {
        // Longest first onto the least loaded shard: every process computes the same split from the same plan.
        var order: std.ArrayList(u32) = .empty;
        const ms = gpa.alloc(u64, kept.items.len) catch panic("out of memory", .{});
        for (kept.items, 0..) |t, i| {
            ms[i] = planned(&plan, t).ms;
            if (isSharded(t)) mine[i] = true else order.append(gpa, @intCast(i)) catch panic("out of memory", .{});
        }
        std.mem.sort(u32, order.items, ByTime{ .ms = ms }, ByTime.longerFirst);
        const load = gpa.alloc(u64, sh.n) catch panic("out of memory", .{});
        @memset(load, 0);
        for (order.items) |i| {
            var least: usize = 0;
            for (load, 0..) |l, s| if (l < load[least]) {
                least = s;
            };
            // A test measured at 0 ms still counts, so a shard of many tiny tests is not taken for an empty one.
            load[least] += @max(ms[i], 1);
            if (least == sh.k) mine[i] = true;
        }
    } else {
        for (kept.items, 0..) |t, i| mine[i] = i % sh.n == sh.k or isSharded(t);
    }
    var len: usize = 0;
    for (kept.items, mine) |t, m| if (m) {
        kept.items[len] = t;
        len += 1;
    };
    tests = kept.items[@min(opt_skip, len)..len];
}

/// The test root's process setup (`src/tests.zig`).
extern fn msv_test_process_init() void;
/// The test root's per-test setup (`src/tests.zig`): every test starts in the test world (`src/sdk/testing/world.zig`).
extern fn msv_test_enter() void;

extern fn mlx_get_peak_memory(res: *usize) c_int;
extern fn mlx_reset_peak_memory() c_int;

fn clockNs(comptime id: std.c.clockid_t) u64 {
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(id, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// `--times-out`: one record per test, appended: its status, wall and CPU microseconds, and MLX's peak bytes in it (more
/// than 0: the test used the GPU).
const Record = struct {
    fd: c_int = -1,
    wall_ns: u64 = 0,
    cpu_ns: u64 = 0,

    fn open() Record {
        const path = opt_times_out orelse return .{};
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true }, @as(c_uint, 0o644));
        if (fd < 0) panic("--times-out: cannot open {s}", .{path});
        return .{ .fd = fd };
    }

    fn begin(r: *Record) void {
        if (r.fd < 0) return;
        _ = mlx_reset_peak_memory();
        r.wall_ns = clockNs(.MONOTONIC);
        r.cpu_ns = clockNs(.PROCESS_CPUTIME_ID);
    }

    fn end(r: *Record, status: []const u8, name: []const u8) void {
        if (r.fd < 0) return;
        const wall = clockNs(.MONOTONIC) -| r.wall_ns;
        const cpu = clockNs(.PROCESS_CPUTIME_ID) -| r.cpu_ns;
        var peak: usize = 0;
        _ = mlx_get_peak_memory(&peak);
        var buf: [256]u8 = undefined;
        const head = std.fmt.bufPrint(&buf, "{s}\t{d}\t{d}\t{d}\t", .{ status, wall / 1000, cpu / 1000, peak }) catch unreachable;
        _ = std.c.write(r.fd, head.ptr, head.len);
        _ = std.c.write(r.fd, name.ptr, name.len);
        _ = std.c.write(r.fd, "\n", 1);
    }
};

fn parseArgs(args: []const [:0]const u8) struct { listen: bool, cache_dir: ?[]const u8 } {
    var listen = false;
    var opt_cache_dir: ?[]const u8 = null;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--listen=-")) {
            listen = true;
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            testing.random_seed = std.fmt.parseUnsigned(u32, arg["--seed=".len..], 0) catch
                @panic("unable to parse --seed command line argument");
        } else if (std.mem.startsWith(u8, arg, "--cache-dir")) {
            opt_cache_dir = arg["--cache-dir=".len..];
        } else if (std.mem.startsWith(u8, arg, "--plan=")) {
            opt_plan = arg["--plan=".len..];
        } else if (std.mem.startsWith(u8, arg, "--lane=")) {
            opt_lane = std.meta.stringToEnum(Lane, arg["--lane=".len..]) orelse panic("--lane: cpu, gpu or serial, got {s}", .{arg});
        } else if (std.mem.startsWith(u8, arg, "--test-filter=")) {
            opt_filters.append(std.heap.page_allocator, arg["--test-filter=".len..]) catch panic("out of memory", .{});
        } else if (std.mem.eql(u8, arg, "--list")) {
            opt_list = true;
        } else if (std.mem.startsWith(u8, arg, "--skip=")) {
            opt_skip = std.fmt.parseUnsigned(usize, arg["--skip=".len..], 10) catch panic("--skip: {s}", .{arg});
        } else if (std.mem.startsWith(u8, arg, "--times-out=")) {
            opt_times_out = arg["--times-out=".len..];
        } else {
            panic("unrecognized command line argument: {s}", .{arg});
        }
    }
    return .{ .listen = listen, .cache_dir = opt_cache_dir };
}

pub fn main(init: std.process.Init.Minimal) void {
    msv_test_process_init();
    @disableInstrumentation();

    if (builtin.cpu.arch.isSpirV()) {
        // SPIR-V needs an special test-runner
        return;
    }

    if (need_simple) {
        tests = builtin.test_functions;
        return mainSimple() catch |err| panic("test failure: {t}", .{err});
    }

    const args = init.args.toSlice(fba.allocator()) catch |err| panic("unable to parse command line args: {t}", .{err});
    const parsed = parseArgs(args);
    selectTests();

    if (opt_list) {
        for (tests) |t| {
            _ = std.c.write(1, t.name.ptr, t.name.len);
            _ = std.c.write(1, "\n", 1);
        }
        return;
    }

    if (builtin.fuzz) {
        const cache_dir = parsed.cache_dir orelse @panic("missing --cache-dir=[path] argument");
        fuzz_abi.fuzzer_init(.fromSlice(cache_dir));
    }

    if (parsed.listen) {
        return mainServer(init) catch |err| panic("internal test runner failure: {t}", .{err});
    } else {
        return mainTerminal(init);
    }
}

fn mainServer(init: std.process.Init.Minimal) !void {
    @disableInstrumentation();
    stdin_reader = .initStreaming(.stdin(), runner_threaded_io, &stdin_buffer);
    stdout_writer = .initStreaming(.stdout(), runner_threaded_io, &stdout_buffer);
    var server: std.zig.Server = .{
        .in = &stdin_reader.interface,
        .out = &stdout_writer.interface,
    };
    try server.serveStringMessage(.zig_version, builtin.zig_version_string);

    while (true) {
        const hdr = try server.receiveMessage();
        switch (hdr.tag) {
            .exit => {
                return std.process.exit(0);
            },
            .query_test_metadata => {
                var sa: std.heap.SafeAllocator = .init(std.heap.page_allocator, .{});
                defer if (sa.deinit() != 0) @panic("internal test runner memory leak");
                const gpa = sa.allocator();

                var string_bytes: std.ArrayList(u8) = .empty;
                defer string_bytes.deinit(gpa);
                try string_bytes.append(gpa, 0); // Reserve 0 for null.

                const test_fns = tests;
                const names = try gpa.alloc(u32, test_fns.len);
                defer gpa.free(names);
                const expected_panic_msgs = try gpa.alloc(u32, test_fns.len);
                defer gpa.free(expected_panic_msgs);

                for (test_fns, names, expected_panic_msgs) |test_fn, *name, *expected_panic_msg| {
                    name.* = @intCast(string_bytes.items.len);
                    try string_bytes.ensureUnusedCapacity(gpa, test_fn.name.len + 1);
                    string_bytes.appendSliceAssumeCapacity(test_fn.name);
                    string_bytes.appendAssumeCapacity(0);
                    expected_panic_msg.* = 0;
                }

                try server.serveTestMetadata(.{
                    .names = names,
                    .expected_panic_msgs = expected_panic_msgs,
                    .string_bytes = string_bytes.items,
                });
            },

            .run_test => {
                testing.environ = init.environ;
                testing.allocator_instance = .init(std.heap.page_allocator, .{
                    .canary = 0xc3a701ba,
                    .check_write_after_free = true,
                });
                testing.io_instance = .init(testing.allocator, .{
                    .argv0 = .init(init.args),
                    .environ = init.environ,
                });
                log_err_count = 0;
                const index = try server.receiveBody_u32();
                const test_fn = tests[index];
                is_fuzz_test = false;

                // let the build server know we're starting the test now
                try server.serveStringMessage(.test_started, &.{});

                const TestResults = std.zig.Server.Message.TestResults;
                msv_test_enter();
                const status: TestResults.Status = if (test_fn.func()) |v| s: {
                    v;
                    break :s .pass;
                } else |err| switch (err) {
                    error.SkipZigTest => .skip,
                    else => s: {
                        if (@errorReturnTrace()) |trace| {
                            std.debug.dumpErrorReturnTrace(trace);
                        }
                        break :s .fail;
                    },
                };
                testing.io_instance.deinit();
                const leak_count = testing.allocator_instance.deinit();
                try server.serveTestResults(.{
                    .index = index,
                    .flags = .{
                        .status = status,
                        .fuzz = is_fuzz_test,
                        .log_err_count = std.math.lossyCast(
                            @FieldType(TestResults.Flags, "log_err_count"),
                            log_err_count,
                        ),
                        .leak_count = std.math.lossyCast(
                            @FieldType(TestResults.Flags, "leak_count"),
                            leak_count,
                        ),
                    },
                });
            },
            .start_fuzzing => {
                // This ensures that this code won't be analyzed and hence reference fuzzer symbols
                // since they are not present.
                if (!builtin.fuzz) unreachable;

                var gpa_instance: std.heap.SafeAllocator = .init(std.heap.page_allocator, .{});
                defer if (gpa_instance.deinit() != 0) {
                    @panic("internal test runner memory leak");
                };
                const gpa = gpa_instance.allocator();
                var io_instance: Io.Threaded = .init(gpa, .{
                    .argv0 = .init(init.args),
                    .environ = init.environ,
                });
                defer io_instance.deinit();

                const mode: fuzz_abi.LimitKind = @fromBackingInt(@intCast(try server.receiveBody_u8()));
                const amount_or_instance = try server.receiveBody_u64();
                const main_instance = mode == .iterations or amount_or_instance == 0;

                if (main_instance) {
                    const coverage = fuzz_abi.fuzzer_coverage();
                    try server.serveCoverageIdMessage(
                        coverage.id,
                        coverage.runs,
                        coverage.unique,
                        coverage.seen,
                    );
                }

                const n_tests: u32 = try server.receiveBody_u32();
                const test_indexes = try gpa.alloc(u32, n_tests);
                defer gpa.free(test_indexes);
                fuzz_runner = .{
                    .indexes = test_indexes,
                    .server = &server,
                    .gpa = gpa,
                    .threaded_io = &io_instance,
                    .input_poller = undefined,
                };

                {
                    var large_name_buf: std.ArrayList(u8) = .empty;
                    defer large_name_buf.deinit(gpa);
                    for (test_indexes) |*i| {
                        const name_len = try server.receiveBody_u32();
                        const name = if (name_len <= server.in.buffer.len)
                            try server.in.take(name_len)
                        else large_name: {
                            try large_name_buf.resize(gpa, name_len);
                            try server.in.readSliceAll(large_name_buf.items);
                            break :large_name large_name_buf.items;
                        };

                        for (0.., tests) |test_i, test_fn| {
                            if (std.mem.eql(u8, name, test_fn.name)) {
                                i.* = @intCast(test_i);
                                break;
                            }
                        } else {
                            panic("fuzz test {s} no longer exists", .{name});
                        }

                        if (main_instance) {
                            const relocated_entry_addr = @intFromPtr(tests[i.*].func);
                            const entry_addr = fuzz_abi.fuzzer_unslide_address(relocated_entry_addr);
                            try server.serveU64Message(.fuzz_start_addr, entry_addr);
                        }
                    }
                }

                fuzz_abi.fuzzer_main(n_tests, testing.random_seed, mode, amount_or_instance);

                assert(mode != .forever);
                std.process.exit(0);
            },

            else => {
                std.debug.print("unsupported message: {x}\n", .{@backingInt(hdr.tag)});
                std.process.exit(1);
            },
        }
    }
}

fn mainTerminal(init: std.process.Init.Minimal) void {
    @disableInstrumentation();
    if (builtin.fuzz) @panic("fuzz test requires server");

    const test_fn_list = tests;
    var ok_count: usize = 0;
    var skip_count: usize = 0;
    var fail_count: usize = 0;
    var fuzz_count: usize = 0;
    const root_node = if (builtin.fuzz) std.Progress.Node.none else std.Progress.start(runner_threaded_io, .{
        .root_name = "Test",
        .estimated_total_items = test_fn_list.len,
    });
    const have_tty = Io.File.stderr().isTty(runner_threaded_io) catch unreachable;

    var leaks: usize = 0;
    var record: Record = .open();
    for (test_fn_list, 0..) |test_fn, i| {
        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        testing.io_instance = .init(testing.allocator, .{
            .argv0 = .init(init.args),
            .environ = init.environ,
        });
        const ok_before = ok_count;
        const skip_before = skip_count;
        const log_err_before = log_err_count;
        record.begin();
        defer {
            testing.io_instance.deinit();
            const leaked = testing.allocator_instance.deinit() != 0;
            if (leaked) leaks += 1;
            const status = if (ok_count == ok_before and skip_count == skip_before) "fail" else if (leaked) "leak" else if (log_err_count != log_err_before) "logerr" else if (skip_count != skip_before) "skip" else "pass";
            record.end(status, test_fn.name);
        }
        testing.log_level = .warn;
        testing.environ = init.environ;

        const test_node = root_node.start(test_fn.name, 0);
        if (!have_tty) {
            std.debug.print("{d}/{d} {s}...", .{ i + 1, test_fn_list.len, test_fn.name });
        }
        is_fuzz_test = false;
        msv_test_enter();
        if (test_fn.func()) |_| {
            ok_count += 1;
            test_node.end();
            if (!have_tty) std.debug.print("OK\n", .{});
        } else |err| switch (err) {
            error.SkipZigTest => {
                skip_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...SKIP\n", .{ i + 1, test_fn_list.len, test_fn.name });
                } else {
                    std.debug.print("SKIP\n", .{});
                }
                test_node.end();
            },
            else => {
                fail_count += 1;
                if (have_tty) {
                    std.debug.print("{d}/{d} {s}...FAIL ({t})\n", .{
                        i + 1, test_fn_list.len, test_fn.name, err,
                    });
                } else {
                    std.debug.print("FAIL ({t})\n", .{err});
                }
                if (@errorReturnTrace()) |trace| {
                    std.debug.dumpErrorReturnTrace(trace);
                }
                test_node.end();
            },
        }
        fuzz_count += @intFromBool(is_fuzz_test);
    }
    root_node.end();
    if (ok_count == test_fn_list.len) {
        std.debug.print("All {d} tests passed.\n", .{ok_count});
    } else {
        std.debug.print("{d} passed; {d} skipped; {d} failed.\n", .{ ok_count, skip_count, fail_count });
    }
    if (log_err_count != 0) {
        std.debug.print("{d} errors were logged.\n", .{log_err_count});
    }
    if (leaks != 0) {
        std.debug.print("{d} tests leaked memory.\n", .{leaks});
    }
    if (fuzz_count != 0) {
        std.debug.print("{d} fuzz tests found.\n", .{fuzz_count});
    }
    if (leaks != 0 or log_err_count != 0 or fail_count != 0) {
        std.process.exit(1);
    }
}

pub fn log(
    comptime message_level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    @disableInstrumentation();
    if (@backingInt(message_level) <= @backingInt(std.log.Level.err)) {
        log_err_count +|= 1;
    }
    if (@backingInt(message_level) <= @backingInt(testing.log_level)) {
        std.debug.print(
            "[" ++ @tagName(scope) ++ "] (" ++ @tagName(message_level) ++ "): " ++ format ++ "\n",
            args,
        );
    }
}

/// Simpler main(), exercising fewer language features, so that
/// work-in-progress backends can handle it.
pub fn mainSimple() anyerror!void {
    @disableInstrumentation();
    // is the backend capable of calling `Io.File.writeAll`?
    const enable_write = switch (builtin.zig_backend) {
        .stage2_aarch64, .stage2_riscv64 => true,
        else => false,
    };
    // is the backend capable of calling `Io.Writer.print`?
    const enable_print = switch (builtin.zig_backend) {
        .stage2_aarch64, .stage2_riscv64 => true,
        else => false,
    };

    testing.allocator_instance = .init(std.heap.page_allocator, .{});
    testing.io_instance = .init(testing.allocator, .{});

    var passed: u64 = 0;
    var skipped: u64 = 0;
    var failed: u64 = 0;

    // we don't want to bring in File and Writer if the backend doesn't support it
    const stdout = if (enable_write) Io.File.stdout() else {};

    for (tests) |test_fn| {
        if (enable_write) {
            stdout.writeStreamingAll(runner_threaded_io, test_fn.name) catch {};
            stdout.writeStreamingAll(runner_threaded_io, "... ") catch {};
        }
        msv_test_enter();
        if (test_fn.func()) |_| {
            if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "PASS\n") catch {};
        } else |err| {
            if (err != error.SkipZigTest) {
                if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "FAIL\n") catch {};
                failed += 1;
                if (!enable_write) return err;
                continue;
            }
            if (enable_write) stdout.writeStreamingAll(runner_threaded_io, "SKIP\n") catch {};
            skipped += 1;
            continue;
        }
        passed += 1;
    }
    if (enable_print) {
        var unbuffered_stdout_writer = stdout.writer(runner_threaded_io, &.{});
        unbuffered_stdout_writer.interface.print(
            "{} passed, {} skipped, {} failed\n",
            .{ passed, skipped, failed },
        ) catch {};
    }
    if (failed != 0) std.process.exit(1);
}

var is_fuzz_test: bool = undefined;
var fuzz_runner: if (builtin.fuzz) struct {
    indexes: []u32,
    server: *std.zig.Server,
    gpa: std.mem.Allocator,
    threaded_io: *Io.Threaded,
    input_poller: Io.Future(Io.Cancelable!void),

    comptime {
        assert(builtin.fuzz); // `fuzz_runner` was analyzed in non-fuzzing compilation
    }

    export fn runner_test_run(i: u32) void {
        @disableInstrumentation();

        fuzz_runner.server.serveU32Message(.fuzz_test_change, i) catch |e| switch (e) {
            error.WriteFailed => panic("failed to write to stdout: {t}", .{stdout_writer.err.?}),
        };

        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        defer if (testing.allocator_instance.deinit() != 0) std.process.exit(1);
        is_fuzz_test = false;

        testing.io_instance = .init(testing.allocator, .{
            .argv0 = fuzz_runner.threaded_io.argv0,
            .environ = fuzz_runner.threaded_io.environ.process_environ,
        });
        defer testing.io_instance.deinit();

        msv_test_enter();
        tests[fuzz_runner.indexes[i]].func() catch |err| switch (err) {
            error.SkipZigTest => return,
            else => {
                if (@errorReturnTrace()) |trace| {
                    std.debug.dumpErrorReturnTrace(trace);
                }
                std.debug.print("failed with error.{t}\n", .{err});
                std.process.exit(1);
            },
        };

        if (!is_fuzz_test) @panic("missed call to std.testing.fuzz");
        if (log_err_count != 0) @panic("error logs detected");
    }

    export fn runner_test_name(i: u32) fuzz_abi.Slice {
        @disableInstrumentation();
        return .fromSlice(tests[fuzz_runner.indexes[i]].name);
    }

    export fn runner_broadcast_input(test_i: u32, bytes_slice: fuzz_abi.Slice) void {
        @disableInstrumentation();
        const bytes = bytes_slice.toSlice();
        fuzz_runner.server.serveBroadcastFuzzInputMessage(test_i, bytes) catch |e| switch (e) {
            error.WriteFailed => panic("failed to write to stdout: {t}", .{stdout_writer.err.?}),
        };
    }

    export fn runner_start_input_poller() void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        const future = io.concurrent(inputPoller, .{}) catch |e| switch (e) {
            error.ConcurrencyUnavailable => @panic("failed to spawn concurrent fuzz input poller"),
        };
        fuzz_runner.input_poller = future;
    }

    export fn runner_stop_input_poller() void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        assert(fuzz_runner.input_poller.cancel(io) == error.Canceled);
    }

    export fn runner_futex_wait(ptr: *const u32, expected: u32) bool {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        return io.futexWait(u32, ptr, expected) == error.Canceled;
    }

    export fn runner_futex_wake(ptr: *const u32, waiters: u32) void {
        @disableInstrumentation();
        const io = fuzz_runner.threaded_io.io();
        io.futexWake(u32, ptr, waiters);
    }

    fn inputPoller() Io.Cancelable!void {
        @disableInstrumentation();
        switch (inputPollerInner()) {
            error.Canceled => |e| return e,
            error.ReadFailed => {
                if (stdin_reader.err.? == error.Canceled) return error.Canceled;
                panic("failed to read from stdin: {t}", .{stdin_reader.err.?});
            },
            error.EndOfStream => @panic("unexpected end of stdin"),
        }
    }

    fn inputPollerInner() (Io.Cancelable || Io.Reader.Error) {
        @disableInstrumentation();
        const server = fuzz_runner.server;
        var large_bytes_list: std.ArrayList(u8) = .empty;
        defer large_bytes_list.deinit(fuzz_runner.gpa);
        while (true) {
            const hdr = try server.receiveMessage();
            if (hdr.tag != .new_fuzz_input) {
                panic("unexpected message: {x}\n", .{@backingInt(hdr.tag)});
            }
            const test_i = try server.receiveBody_u32();
            const input_len = hdr.bytes_len - 4;
            const bytes = if (input_len <= server.in.buffer.len)
                try server.in.take(input_len)
            else bytes: {
                large_bytes_list.resize(fuzz_runner.gpa, @intCast(input_len)) catch @panic("OOM");
                try server.in.readSliceAll(large_bytes_list.items);
                break :bytes large_bytes_list.items;
            };
            if (fuzz_abi.fuzzer_receive_input(test_i, .fromSlice(bytes))) {
                return error.Canceled;
            }
        }
    }
} else void = undefined;

pub fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), *std.testing.Smith) anyerror!void,
    options: testing.FuzzInputOptions,
) anyerror!void {
    // Prevent this function from confusing the fuzzer by omitting its own code
    // coverage from being considered.
    @disableInstrumentation();

    // Some compiler backends are not capable of handling fuzz testing yet but
    // we still want CI test coverage enabled.
    if (need_simple) return;

    // Smoke test to ensure the test did not use conditional compilation to
    // contradict itself by making it not actually be a fuzz test when the test
    // is built in fuzz mode.
    is_fuzz_test = true;

    // Ensure no test failure occurred before starting fuzzing.
    if (log_err_count != 0) @panic("error logs detected");

    // libfuzzer is in a separate compilation unit so that its own code can be
    // excluded from code coverage instrumentation. It needs a function pointer
    // it can call for checking exactly one input. Inside this function we do
    // our standard unit test checks such as memory leaks, and interaction with
    // error logs.
    const global = struct {
        var ctx: @TypeOf(context) = undefined;

        fn test_one() callconv(.c) bool {
            @disableInstrumentation();
            testing.allocator_instance = .init(std.heap.page_allocator, .{
                .canary = 0xcacce5e0,
                .check_write_after_free = true,
            });
            defer if (testing.allocator_instance.deinit() != 0) std.process.exit(1);
            log_err_count = 0;
            testOne(ctx, @constCast(&testing.Smith{ .in = null })) catch |err| switch (err) {
                error.SkipZigTest => return true,
                else => {
                    const stderr = std.debug.lockStderr(&.{}).terminal();
                    p: {
                        if (@errorReturnTrace()) |trace| {
                            std.debug.writeErrorReturnTrace(trace, stderr) catch break :p;
                        }
                        stderr.writer.print("failed with error.{t}\n", .{err}) catch break :p;
                    }
                    std.process.exit(1);
                },
            };
            if (log_err_count != 0) {
                const stderr = std.debug.lockStderr(&.{}).terminal();
                stderr.writer.print("error logs detected\n", .{}) catch {};
                std.process.exit(1);
            }
            return false;
        }
    };

    if (builtin.fuzz) {
        // Preserve the calling test's allocator state
        const prev_allocator_state = testing.allocator_instance;
        defer testing.allocator_instance = prev_allocator_state;

        global.ctx = context;
        fuzz_abi.fuzzer_set_test(&global.test_one);
        for (options.corpus) |elem|
            fuzz_abi.fuzzer_new_input(.fromSlice(elem));
        fuzz_abi.fuzzer_start_test();
        return;
    }

    // When the unit test executable is not built in fuzz mode, only run the
    // provided corpus.
    for (options.corpus) |input| {
        var smith: testing.Smith = .{ .in = input };
        try testOne(context, &smith);
    }

    // In case there is no provided corpus, also use an empty
    // string as a smoke test.
    var smith: testing.Smith = .{ .in = "" };
    try testOne(context, &smith);
}
