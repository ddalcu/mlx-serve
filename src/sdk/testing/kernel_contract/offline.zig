//! M1 and M2 thread limits without those chips, for gate 5's records: every registered launch's source as MLX
//! generates it, at every shape its spec runs, compiled with Xcode's Metal toolchain for M1 and M2 (`metal-tt`);
//! native code built for T threads equal to the plain build means the plain pipeline runs T. Off unless
//! `MLX_SERVE_KERNEL_CONTRACT_OFFLINE` names a scratch directory (it needs the Metal device to generate sources).

const std = @import("std");
const env = @import("../../../env.zig");
const builtin = @import("builtin");
const mlx = @import("../../../mlx/ffi.zig");
const spec = @import("spec.zig");
const registry = @import("registry.zig");

/// The translator's M1 and M2 targets (phones left out); a family's limit is its lowest.
const m1_archs = [_][]const u8{ "applegpu_g13g", "applegpu_g13s", "applegpu_g13d" };
const m2_archs = [_][]const u8{ "applegpu_g14g", "applegpu_g14s", "applegpu_g14d" };

const Ctx = struct { a: std.mem.Allocator, io: std.Io, dir: []const u8, os_major: u32 };

/// The running macOS major version (MLX picks its Metal language version by it).
fn osMajor() u32 {
    var buf: [32]u8 = undefined;
    var len: usize = buf.len;
    if (std.c.sysctlbyname("kern.osproductversion", &buf, &len, null, 0) != 0) return 26;
    const text = std.mem.sliceTo(buf[0..len], 0);
    const dot = std.mem.indexOfScalar(u8, text, '.') orelse text.len;
    return std.fmt.parseInt(u32, text[0..dot], 10) catch 26;
}

/// MLX's language version for this macOS, as its device picks it.
fn stdFlag(major: u32) []const u8 {
    return if (major >= 27) "-std=metal4.1" else if (major >= 26) "-std=metal4.0" else if (major >= 15) "-std=metal3.2" else "-std=metal3.1";
}

fn exec(x: Ctx, argv: []const []const u8) !void {
    const r = try std.process.run(x.a, x.io, .{ .argv = argv });
    defer x.a.free(r.stdout);
    defer x.a.free(r.stderr);
    if (!r.term.success()) {
        std.debug.print("{s} failed: {s}\n", .{ argv[2], r.stderr });
        return error.ToolFailed;
    }
}

/// The source compiled as MLX compiles it at run time, into a metallib named by its hash.
fn library(x: Ctx, source: []const u8) ![]const u8 {
    var h = std.hash.Wyhash.init(0);
    h.update(source);
    const stem = try std.fmt.allocPrint(x.a, "{s}/src_{x}", .{ x.dir, h.final() });
    const lib = try std.fmt.allocPrint(x.a, "{s}.metallib", .{stem});
    if (std.Io.Dir.cwd().access(x.io, lib, .{})) |_| return lib else |_| {}
    const src = try std.fmt.allocPrint(x.a, "{s}.metal", .{stem});
    const air = try std.fmt.allocPrint(x.a, "{s}.air", .{stem});
    // MLX prepends its `metal::utils()` preamble, generated from this header of the pinned submodule.
    const full = try std.fmt.allocPrint(x.a, "#include \"mlx/backend/metal/kernels/utils.h\"\n{s}", .{source});
    try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = src, .data = full });
    try exec(x, &.{ "xcrun", "-sdk", "macosx", "metal", stdFlag(x.os_major), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast", "-I", "lib/mlx-src", "-c", src, "-o", air });
    try exec(x, &.{ "xcrun", "-sdk", "macosx", "metallib", air, "-o", lib });
    return lib;
}

/// A section's bytes in a Mach-O image at `base`, found by section name.
fn section(buf: []const u8, base: usize, name: []const u8) ![]const u8 {
    if (buf.len < base + 32 or std.mem.readInt(u32, buf[base..][0..4], .little) != 0xfeedfacf) return error.NotMachO;
    const ncmds = std.mem.readInt(u32, buf[base + 16 ..][0..4], .little);
    var at = base + 32;
    for (0..ncmds) |_| {
        const cmd = std.mem.readInt(u32, buf[at..][0..4], .little);
        const size = std.mem.readInt(u32, buf[at + 4 ..][0..4], .little);
        if (cmd == 0x19) for (0..std.mem.readInt(u32, buf[at + 64 ..][0..4], .little)) |i| {
            const s = buf[at + 72 + 80 * i ..][0..80];
            if (std.mem.eql(u8, std.mem.sliceTo(s[0..16], 0), name)) {
                const len: usize = @intCast(std.mem.readInt(u64, s[40..48], .little));
                const off: usize = std.mem.readInt(u32, s[48..52], .little);
                return buf[base + off ..][0..len];
            }
        };
        at += size;
    }
    return error.NoSection;
}

/// The native code metal-tt builds for one architecture, asking for `threads` a group (0: no request).
fn native(x: Ctx, lib: []const u8, fun: []const u8, arch: []const u8, threads: u32) ![]const u8 {
    const out = try std.fmt.allocPrint(x.a, "{s}.{x}.{s}.{d}.bin", .{ lib, std.hash.Wyhash.hash(0, fun), arch, threads });
    if (std.Io.Dir.cwd().access(x.io, out, .{})) |_| {} else |_| {
        var script: std.ArrayList(u8) = .empty;
        try script.print(x.a, "{{\"libraries\": {{\"paths\": [{{\"path\": \"{s}\", \"label\": \"lib\"}}]}}, \"pipelines\": {{\"compute_pipelines\": [{{\"compute_function\": \"alias:lib#{s}\"", .{ lib, fun });
        if (threads > 0) try script.print(x.a, ", \"max_total_threads_per_threadgroup\": {d}", .{threads});
        try script.appendSlice(x.a, "}]}}");
        const path = try std.fmt.allocPrint(x.a, "{s}.mtlp-json", .{out});
        try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = path, .data = script.items });
        const ver = try std.fmt.allocPrint(x.a, "{d}.0", .{x.os_major});
        try exec(x, &.{ "xcrun", "metal-tt", "-arch", arch, "-macos-version-min", ver, "-o", out, lib, path });
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(x.io, out, x.a, .limited(1 << 26));
    const outer = try section(bytes, 0, "__compute");
    const inner_at = @intFromPtr(outer.ptr) - @intFromPtr(bytes.ptr);
    return section(bytes, inner_at, "__text");
}

/// The most threads (a multiple of 32) whose build equals the plain one: what the plain pipeline runs at.
fn limitOn(x: Ctx, lib: []const u8, fun: []const u8, arch: []const u8) !u32 {
    const plain = try native(x, lib, fun, arch, 0);
    if (std.mem.eql(u8, plain, try native(x, lib, fun, arch, 1024))) return 1024;
    var lo: u32 = 0;
    var hi: u32 = 33;
    while (lo + 1 < hi) {
        const mid = (lo + hi) / 2;
        if (std.mem.eql(u8, plain, try native(x, lib, fun, arch, 32 * mid))) lo = mid else hi = mid;
    }
    return 32 * lo;
}

fn familyLimit(x: Ctx, lib: []const u8, fun: []const u8, archs: []const []const u8) !u32 {
    var least: u32 = 1024;
    for (archs) |arch| least = @min(least, try limitOn(x, lib, fun, arch));
    return least;
}

/// The function MLX compiles: the template instantiation's host name, else the kernel itself.
fn functionName(source: []const u8) ?[]const u8 {
    if (std.mem.indexOf(u8, source, "[[host_name(\"")) |at| {
        const from = at + "[[host_name(\"".len;
        const end = std.mem.indexOfScalarPos(u8, source, from, '"') orelse return null;
        return source[from..end];
    }
    const key = "[[kernel]] void ";
    const at = std.mem.indexOf(u8, source, key) orelse return null;
    const end = std.mem.indexOfScalarPos(u8, source, at + key.len, '(') orelse return null;
    return source[at + key.len .. end];
}

// ── tests ──

test "kernel contract offline: each record's M1/M2 limit is what the toolchain compiles" {
    const dir = env.get(.MLX_SERVE_KERNEL_CONTRACT_OFFLINE) orelse return error.SkipZigTest;
    if (!builtin.os.tag.isDarwin() or mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const x = Ctx{ .a = a, .io = std.testing.io, .dir = std.mem.span(dir), .os_major = osMajor() };
    try std.Io.Dir.cwd().createDirPath(x.io, x.dir);
    var set = try registry.build(std.testing.allocator);
    defer set.deinit();
    var bad: usize = 0;
    for (set.kernels.items) |k| {
        const launch_of = k.vt.launch orelse continue;
        const l = launch_of(k.ctx);
        if (!l.offline) continue;
        const shapes = try spec.shapes(a, k.spec);
        for (shapes) |shape| {
            const g = k.vt.geometry.?(k.ctx, shape) orelse continue;
            if (!try k.vt.prepare(k.ctx, shape, .random, 1)) continue;
            const rig = k.vt.rig(k.ctx);
            var src: std.ArrayList(u8) = .empty;
            defer src.deinit(rig.a);
            rig.capture = &src;
            try k.vt.run(k.ctx);
            const fun = functionName(src.items) orelse return error.NoKernelName;
            const lib = try library(x, src.items);
            const m = [2]u32{ try familyLimit(x, lib, fun, &m1_archs), try familyLimit(x, lib, fun, &m2_archs) };
            const rec = l.limitsAt(shape);
            const at = .{ k.spec.name, shape.rows, shape.n, shape.k, g.threads(), m[0], m[1], rec[0], rec[1] };
            if (m[0] == 0 or m[1] == 0) {
                // Every thread request changed the build: equality cannot tell its limit.
                std.debug.print("INCONCLUSIVE offline {s} at {d}x{d}x{d}: {d} threads; no requested build equals the plain one (recorded {d}, {d})\n", .{ k.spec.name, shape.rows, shape.n, shape.k, g.threads(), rec[0], rec[1] });
                continue;
            }
            // A record is the least any toolchain compiled: never above what compiled here, never unrecorded.
            const current = rec[0] != 0 and rec[0] <= m[0] and rec[1] <= m[1];
            if (!current) bad += 1;
            if (!current or m[0] < 1024 or m[1] < 1024) std.debug.print("{s} offline {s} at {d}x{d}x{d}: {d} threads; M1 pipelines take {d}, M2 {d} (recorded {d}, {d})\n", .{if (current) "PASS" else "FAIL"} ++ at);
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}
