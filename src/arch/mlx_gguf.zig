//! Bridge to lib/mlx-serve-gguf: GGUF files served on the regular MLX path.
//! The file stands in for a model dir: config.json, tokenizer.json,
//! tokenizer_config.json and generation_config.json are rebuilt from its
//! metadata, and its tensors load as raw ggml blocks that
//! `Transformer.qmatmul` runs through custom kernels.
//! Anything the module can't serve stays with the embedded engines.
const std = @import("std");
const gguf = @import("mlx_serve_gguf");
const log = @import("../log.zig");
const mlx = @import("../mlx.zig");
const model_discovery = @import("../model_discovery.zig");

pub const kernels = gguf.kernels;

/// Opt-in (`--mlx-gguf`); `--engine ds4|llama` keeps it off.
pub var enabled: bool = false;

/// Resolved .gguf path when this engine serves `model_dir`, else null. Caller frees.
pub fn servablePath(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    if (!enabled or !model_discovery.isGgufModelPath(io, model_dir)) return null;
    const path = model_discovery.resolveGgufFile(io, allocator, model_dir) catch return null;
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

test "real GGUF: stands in for a model dir, and --engine turns it off (set MLX_SERVE_GGUF_TEST_MODEL)" {
    const model = std.mem.span(std.c.getenv("MLX_SERVE_GGUF_TEST_MODEL") orelse return error.SkipZigTest);
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const model_mod = @import("../model.zig");
    const tokenizer_mod = @import("../tokenizer.zig");

    var config = try model_mod.parseConfig(io, allocator, model);
    defer config.deinit(allocator);
    try std.testing.expectEqual(model_mod.QuantMode.gguf, config.quant_mode);
    try std.testing.expect(config.num_hidden_layers > 0 and config.quant_bits > 0);

    var tok = try tokenizer_mod.loadTokenizer(io, allocator, model);
    defer tok.deinit();
    const text = "Hello, GGUF on MLX! 12345 héllo";
    const ids = try tok.encode(allocator, text);
    defer allocator.free(ids);
    const back = try tok.decode(allocator, ids, false);
    defer allocator.free(back);
    try std.testing.expectEqualStrings(text, back);
    try std.testing.expect(tok.special_tokens.count() > 0);

    try std.testing.expect(weightBytes(io, model).? > 1 << 20);
    enabled = false;
    defer enabled = true;
    try std.testing.expectEqual(@as(?[]u8, null), servablePath(io, allocator, model));
}
