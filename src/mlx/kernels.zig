//! Kernel text. Every Metal and CUDA source lives in a file under `src/`: `kernels/<area>/` when shared,
//! `plugins/<name>/kernels/` when one model's, named by its path from `src/` and listed with its sha256 in
//! `kernels/digests.txt` (`shasum -a 256` format, run from `src/`). A `.fmt.metal` / `.fmt.cu` file is a
//! `std.fmt` template: braces doubled, each `{s}` / `{d}` filled by its wrapper.
//! `source` embeds a file at compile time. `resolve` (every kernel MLX compiles passes through it, from
//! `ffi.zig`) hands MLX the copy under `MLX_SERVE_KERNEL_DIR` (a tree laid out like `src/`) of any text that
//! is one whole file: edit a kernel, restart. Pieces and templates assembled in Zig need a rebuild.

const std = @import("std");
const env = @import("../env.zig");
const crt = @import("../sdk/util/crt.zig");
const log = @import("../sdk/util/log.zig");

/// The file's bytes, comptime-known and typed like a string literal, so `++` and `std.fmt` compose them.
pub inline fn source(comptime path: []const u8) @TypeOf(@embedFile("../" ++ path)) {
    return @embedFile("../" ++ path);
}

const File = struct { path: []const u8, digest: []const u8, text: [:0]const u8 };

const files: []const File = blk: {
    @setEvalBranchQuota(1_000_000);
    var out: []const File = &.{};
    var lines = std.mem.tokenizeScalar(u8, @embedFile("../kernels/digests.txt"), '\n');
    while (lines.next()) |line| out = out ++ [_]File{.{ .path = line[66..], .digest = line[0..64], .text = source(line[66..]) }};
    const final = out[0..out.len].*;
    break :blk &final;
};

/// Texts read from one override directory, each at its first use; a file it cannot read keeps the built-in text.
const Overrides = struct {
    root: []const u8,
    texts: [files.len]?[:0]const u8 = @splat(null),

    fn get(self: *Overrides, text: [*:0]const u8) [*:0]const u8 {
        const t = std.mem.span(text);
        for (files, &self.texts) |f, *slot| {
            if (!std.mem.eql(u8, f.text, t)) continue;
            if (slot.* == null) slot.* = read(self.root, f.path) orelse f.text;
            return slot.*.?.ptr;
        }
        return text;
    }

    fn read(root: []const u8, path: []const u8) ?[:0]const u8 {
        var buf: [4096]u8 = undefined;
        const full = std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ root, path }, 0) catch return null;
        const fd = crt.openRead(full);
        if (fd < 0) {
            log.warn("[kernels] {s} not readable, keeping the built-in text\n", .{full});
            return null;
        }
        defer crt.close(fd);
        const n = crt.seekEnd(fd);
        if (n < 0) return null;
        const out = std.heap.c_allocator.allocSentinel(u8, @intCast(n), 0) catch return null;
        if (crt.pread(fd, out.ptr, out.len, 0) != n) {
            std.heap.c_allocator.free(out);
            return null;
        }
        log.info("[kernels] {s}\n", .{full});
        return out;
    }
};

var overrides: ?Overrides = null;
var env_read = false;

/// `text` as MLX should compile it: itself, or with `MLX_SERVE_KERNEL_DIR` set and `text` one whole kernel
/// file, that file from the directory. Kernels are created on the inference thread only.
pub fn resolve(text: [*:0]const u8) [*:0]const u8 {
    if (!env_read) {
        env_read = true;
        if (env.get(.MLX_SERVE_KERNEL_DIR)) |dir| overrides = .{ .root = std.mem.span(dir) };
    }
    return if (overrides) |*o| o.get(text) else text;
}

// ── tests ──

test "every kernel file still has the bytes digests.txt records" {
    for (files) |f| {
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(f.text, &h, .{});
        const hex = std.fmt.bytesToHex(h, .lower);
        if (!std.mem.eql(u8, &hex, f.digest)) {
            std.debug.print("kernel text changed; its digests.txt line is now:\n{s}  {s}\n", .{ &hex, f.path });
            return error.KernelDigestMismatch;
        }
    }
}

test "every .metal and .cu file under src/ is in digests.txt" {
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, "src", .{ .iterate = true });
    defer dir.close(io);
    var walker = try dir.walk(std.testing.allocator);
    defer walker.deinit();
    while (try walker.next(io)) |e| {
        if (e.kind != .file) continue;
        if (!std.mem.endsWith(u8, e.path, ".metal") and !std.mem.endsWith(u8, e.path, ".cu")) continue;
        for (files) |f| {
            if (std.mem.eql(u8, f.path, e.path)) break;
        } else {
            std.debug.print("src/{s} is missing from src/kernels/digests.txt\n", .{e.path});
            return error.KernelFileUnlisted;
        }
    }
}

test "an override directory replaces whole-file kernel texts and nothing else" {
    const root = ".zig-cache/tmp/kernels-override-test";
    crt.mkdir(".zig-cache/tmp");
    crt.mkdir(root);
    crt.mkdir(root ++ "/kernels");
    crt.mkdir(root ++ "/kernels/common");
    const edited = "#include <metal_tensor>\n";
    const fd = crt.openWrite(root ++ "/kernels/common/mpp_tensor_ops.metal", true);
    try std.testing.expect(fd >= 0);
    _ = crt.write(fd, edited);
    crt.close(fd);
    defer crt.unlink(root ++ "/kernels/common/mpp_tensor_ops.metal");

    var o: Overrides = .{ .root = root };
    try std.testing.expectEqualStrings(edited, std.mem.span(o.get(source("kernels/common/mpp_tensor_ops.metal"))));
    const absent = source("kernels/common/mpp_tensor.metal");
    try std.testing.expectEqual(@as([*:0]const u8, absent), o.get(absent));
    const piece = source("kernels/common/mpp_tensor_ops.metal") ++ "// more";
    try std.testing.expectEqual(@as([*:0]const u8, piece), o.get(piece));
}
