//! GGUF files served on the regular MLX path.
//! The file stands in for a model dir: config.json, tokenizer.json,
//! tokenizer_config.json and generation_config.json are rebuilt from its
//! metadata, and its tensors load as raw ggml blocks that
//! `Transformer.qmatmul` runs through custom kernels.
//! Anything the module can't serve stays with the embedded engines.
const std = @import("std");
const gguf = @import("root.zig");
const log = @import("../../util/log.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gguf_files = @import("files.zig");

pub const kernels = gguf.kernels;

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ClaimState = struct {
    /// Opt-in (`--mlx-gguf`); `--engine ds4|llama` keeps it off.
    enabled: bool = false,
};
pub const claimState_shared = @import("../../util/shared.zig").Shared(ClaimState, "gguf_claim", .{});
pub inline fn claimState() *ClaimState {
    return claimState_shared.get();
}

/// Resolved .gguf path when this engine serves `model_dir`, else null. Caller frees.
pub fn servablePath(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    if (comptime @import("builtin").os.tag == .windows) return null; // Metal-only engine
    if (!claimState().enabled or !gguf_files.isGgufModelPath(io, model_dir)) return null;
    const path = gguf_files.resolveGgufFile(io, allocator, model_dir) catch return null;
    var f = gguf.gguf.File.open(allocator, path) catch {
        allocator.free(path);
        return null;
    };
    defer f.deinit();
    var buf: [256]u8 = undefined;
    if (gguf.meta.unsupportedReason(&f, &buf)) |why| {
        log.debug("[gguf] mlx engine declines {s}: {s}\n", .{ path, why });
        allocator.free(path);
        return null;
    }
    return path;
}

pub const Sidecar = enum { config, tokenizer, tokenizer_config, generation_config };

/// The JSON a model dir would hold in `<which>.json`, null when `model_dir`
/// is not ours. Caller frees.
pub fn sidecar(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, which: Sidecar) !?[]u8 {
    if (comptime @import("builtin").os.tag == .windows) return null; // Metal-only engine
    const path = servablePath(io, allocator, model_dir) orelse return null;
    defer allocator.free(path);
    var f = try gguf.gguf.File.open(allocator, path);
    defer f.deinit();
    return switch (which) {
        .config => try gguf.meta.configJson(allocator, &f),
        .tokenizer => try gguf.meta.tokenizerJson(allocator, &f),
        .tokenizer_config => try gguf.meta.tokenizerConfigJson(allocator, &f),
        .generation_config => try gguf.meta.generationConfigJson(allocator, &f),
    };
}

/// Size of the GGUF this engine would load for `model_dir` (what ends up
/// resident: tensors are copied as is), null when `model_dir` is not ours.
pub fn weightBytes(io: std.Io, model_dir: []const u8) ?u64 {
    const allocator = std.heap.page_allocator;
    const path = servablePath(io, allocator, model_dir) orelse return null;
    defer allocator.free(path);
    const file = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer file.close(io);
    return (file.stat(io) catch return null).size;
}

/// Fill `out` from the GGUF. False when `model_dir` is not ours.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, out: *std.StringHashMap(mlx.mlx_array)) !bool {
    if (comptime @import("builtin").os.tag == .windows) return false; // Metal-only engine
    const path = servablePath(io, allocator, model_dir) orelse return false;
    defer allocator.free(path);
    var f = try gguf.gguf.File.open(allocator, path);
    defer f.deinit();
    try gguf.weights.load(allocator, &f, out, mlx.gpuStream());
    try gguf.weights.warmKernels(allocator, out, mlx.gpuStream());
    log.info("[gguf] mlx engine: {d} tensors from {s}\n", .{ f.tensors.count(), path });
    return true;
}

test "declines everything that is not a servable GGUF" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "mlx-pack");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx-pack/config.json", .data = "{}" });
    try tmp.dir.createDirPath(io, "junk");
    try tmp.dir.writeFile(io, .{ .sub_path = "junk/model.gguf", .data = "not a gguf" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    for ([_][]const u8{ "mlx-pack", "junk", "junk/model.gguf", "missing" }) |sub| {
        const dir = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, sub });
        defer allocator.free(dir);
        try std.testing.expectEqual(@as(?[]u8, null), servablePath(io, allocator, dir));
        try std.testing.expectEqual(@as(?[]u8, null), try sidecar(io, allocator, dir, .config));
        try std.testing.expectEqual(@as(?u64, null), weightBytes(io, dir));
        var map = std.StringHashMap(mlx.mlx_array).init(allocator);
        defer map.deinit();
        try std.testing.expect(!try loadWeights(io, allocator, dir, &map));
    }
}

test {
    _ = gguf;
}
