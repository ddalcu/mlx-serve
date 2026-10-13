//! MLX's Metal kernel headers, embedded for the kernels that reuse its tiles in their own
//! `metal_kernel` source (src/lane_qmm.zig, src/jangtq2.zig). MLX compiles that source with no
//! include path, so `inlined` puts each header where its `#include "mlx/..."` line was.
const std = @import("std");

const Allocator = std.mem.Allocator;

const Header = struct { path: []const u8, src: []const u8 };
/// Every file reachable from utils.h, steel/gemm/{gemm,nax,loader}.h and quantized{,_utils,_nax}.h.
const kernel_headers = [_]Header{
    header("utils.h"),
    header("bf16.h"),
    header("bf16_math.h"),
    header("complex.h"),
    header("defines.h"),
    header("logging.h"),
    header("steel/gemm/gemm.h"),
    header("steel/gemm/loader.h"),
    header("steel/defines.h"),
    header("steel/gemm/mma.h"),
    header("steel/gemm/transforms.h"),
    header("steel/utils.h"),
    header("steel/utils/integral_constant.h"),
    header("steel/utils/type_traits.h"),
    header("steel/gemm/params.h"),
    header("steel/gemm/nax.h"),
    header("quantized_nax.h"),
    header("quantized_utils.h"),
    header("quantized.h"),
};

fn header(comptime rel: []const u8) Header {
    return .{ .path = "mlx/backend/metal/kernels/" ++ rel, .src = @embedFile("mlx-src/mlx/backend/metal/kernels/" ++ rel) };
}

/// vMLX's kernels.py _mlx_headers, allocated from arena `a`: each file inlined in order, an
/// `#include "mlx/..."` line replaced by that header's own expansion (empty once seen), `#pragma once`
/// dropped. utils.h and everything it includes count as seen: the metal_kernel preamble holds them.
pub fn inlined(a: Allocator, files: []const []const u8) ![]u8 {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var skipped: std.ArrayList(u8) = .empty;
    try expandHeader(a, &skipped, "mlx/backend/metal/kernels/utils.h", &seen);
    var out: std.ArrayList(u8) = .empty;
    for (files) |f| {
        try expandHeader(a, &out, try a.print("mlx/backend/metal/kernels/{s}", .{f}), &seen);
        try out.append(a, '\n');
    }
    return out.items;
}

/// kernels.py _expand: the file's lines (Python splitlines) joined by '\n', without a final newline.
fn expandHeader(a: Allocator, out: *std.ArrayList(u8), rel: []const u8, seen: *std.StringHashMapUnmanaged(void)) !void {
    if (seen.contains(rel)) return;
    try seen.put(a, rel, {});
    const src = for (kernel_headers) |h| {
        if (std.mem.eql(u8, h.path, rel)) break h.src;
    } else return error.MlxHeaderMissing;
    const body = if (std.mem.endsWith(u8, src, "\n")) src[0 .. src.len - 1] else src;
    var lines = std.mem.splitScalar(u8, body, '\n');
    var first = true;
    while (lines.next()) |line| {
        if (std.mem.eql(u8, std.mem.trim(u8, line, " \t\r\x0b\x0c"), "#pragma once")) continue;
        if (!first) try out.append(a, '\n');
        first = false;
        if (includedPath(line)) |inc| try expandHeader(a, out, inc, seen) else try out.appendSlice(a, line);
    }
}

/// The path of a `\s*#include\s+"(mlx/[^"]+)"` line.
fn includedPath(line: []const u8) ?[]const u8 {
    const ws = " \t\r\x0b\x0c";
    const directive = std.mem.trimStart(u8, line, ws);
    if (!std.mem.startsWith(u8, directive, "#include")) return null;
    const after = directive["#include".len..];
    const quoted = std.mem.trimStart(u8, after, ws);
    if (quoted.len == after.len or !std.mem.startsWith(u8, quoted, "\"mlx/")) return null;
    const end = std.mem.indexOfScalarPos(u8, quoted, 1, '"') orelse return null;
    if (end == "\"mlx/".len) return null;
    return quoted[1..end];
}
