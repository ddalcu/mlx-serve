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
    /// Test hook: off, every tensor keeps its ggml blocks and every A_log the file's dtype, the weights the GGUF-file
    /// proofs' tapes were recorded over (`plugins/lfm2/proof.zig` `openGguf`).
    mlx_layouts: bool = true,
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
        .config => try configJson(io, allocator, &f, path),
        .tokenizer => try gguf.meta.tokenizerJson(allocator, &f),
        .tokenizer_config => try gguf.meta.tokenizerConfigJson(allocator, &f),
        .generation_config => try gguf.meta.generationConfigJson(allocator, &f),
    };
}

/// The set's config.json; a DeepSeek-V4.1 set's carries the DSpark keys of the stage bundle and the tower of the
/// vision GGUF beside it.
fn configJson(io: std.Io, allocator: std.mem.Allocator, f: *const gguf.gguf.File, path: []const u8) ![]u8 {
    const arch = try gguf.gguf_arch.Arch.read(f);
    if (arch != .deepseek41) return gguf.meta.configJson(allocator, f);
    const ds41 = gguf.gguf_arch.deepseek41;
    var ds: ?ds41.Dspark = null;
    if (mtpHead(io, allocator, f, path)) |head| {
        defer allocator.free(head);
        var h = try gguf.gguf.File.open(allocator, head);
        defer h.deinit();
        ds = try ds41.Dspark.read(&h, arch.deepseek41);
    }
    var vision: ?ds41.Vision = null;
    if (visionTower(io, allocator, f, path)) |tower| {
        defer allocator.free(tower);
        var v = try gguf.gguf.File.open(allocator, tower);
        defer v.deinit();
        vision = try ds41.Vision.read(&v, arch.deepseek41, null);
    }
    return ds41.configJsonWith(allocator, arch.deepseek41, ds, vision);
}

/// Bytes of the tensors this engine would load for `model_dir` (what ends up
/// resident: tensors are copied as is, skipped ones stay on disk), null when
/// `model_dir` is not ours.
pub fn weightBytes(io: std.Io, model_dir: []const u8) ?u64 {
    const allocator = std.heap.page_allocator;
    const path = servablePath(io, allocator, model_dir) orelse return null;
    defer allocator.free(path);
    var f = gguf.gguf.File.open(allocator, path) catch return null;
    defer f.deinit();
    var bytes = (gguf.weights.residentBytes(&f) catch return null) + gguf.weights.qwen35MtpBytes(&f);
    if (mtpHead(io, allocator, &f, path)) |head| {
        defer allocator.free(head);
        var h = gguf.gguf.File.open(allocator, head) catch return bytes;
        defer h.deinit();
        bytes += switch (gguf.gguf_arch.Arch.read(&f) catch return bytes) {
            .deepseek41 => |hp| gguf.weights.dsparkResidentBytes(&h, hp) catch 0,
            else => gguf.weights.mtpResidentBytes(&h) catch 0,
        };
    }
    // Billed whether or not the load takes it, as a pack's safetensors bill its tower.
    if (visionTower(io, allocator, &f, path)) |tower| {
        defer allocator.free(tower);
        var v = gguf.gguf.File.open(allocator, tower) catch return bytes;
        defer v.deinit();
        bytes += gguf.weights.visionResidentBytes(&v, (gguf.gguf_arch.Arch.read(&f) catch return bytes).deepseek41) catch 0;
    }
    return bytes;
}

/// The vision tower GGUF beside the DeepSeek-V4.1 set at `path` (`gguf_files.findVisionSidecar`) when it was built
/// for this set (`deepseek41.Vision.read`). Caller frees.
fn visionTower(io: std.Io, allocator: std.mem.Allocator, f: *const gguf.gguf.File, path: []const u8) ?[]u8 {
    const trunk = gguf.gguf_arch.Arch.read(f) catch return null;
    if (trunk != .deepseek41) return null;
    const tower = gguf_files.findVisionSidecar(io, allocator, path) orelse return null;
    var v = gguf.gguf.File.open(allocator, tower) catch {
        allocator.free(tower);
        return null;
    };
    defer v.deinit();
    if (gguf.gguf_arch.deepseek41.Vision.read(&v, trunk.deepseek41, f.getString("general.source.revision"))) |_| return tower else |e|
        log.info("[gguf] mlx engine: {s} is not this set's vision tower ({s})\n", .{ std.fs.path.basename(tower), @errorName(e) });
    allocator.free(tower);
    return null;
}

/// The draft GGUF beside the model at `path` (`gguf_files.findMtpSidecar`) when its trunk drafts with one and the
/// file is that kind: a qwen4exp MTP head, or a DeepSeek-V4.1 DSpark stage bundle. Caller frees.
fn mtpHead(io: std.Io, allocator: std.mem.Allocator, f: *const gguf.gguf.File, path: []const u8) ?[]u8 {
    const trunk = gguf.gguf_arch.Arch.read(f) catch return null;
    if (trunk != .qwen4exp and trunk != .deepseek41) return null;
    const head = gguf_files.findMtpSidecar(io, allocator, path) orelse return null;
    var h = gguf.gguf.File.open(allocator, head) catch {
        allocator.free(head);
        return null;
    };
    defer h.deinit();
    if (draftsWith(trunk, &h)) return head;
    log.warn("[gguf] {s} is not this set's draft head: loading without drafting\n", .{head});
    allocator.free(head);
    return null;
}

/// Whether `h` is a draft head `trunk` loads: Qwen3.8 Flash Next's MTP head by arch name, a DeepSeek-V4.1 DSpark
/// bundle only when its stages fit this trunk (a mismatched one would otherwise fail the whole load).
fn draftsWith(trunk: gguf.gguf_arch.Arch, h: *const gguf.gguf.File) bool {
    const name = h.getString("general.architecture") orelse "";
    return switch (trunk) {
        .qwen4exp => std.mem.eql(u8, name, gguf.gguf_arch.qwen4exp.arch_name),
        .deepseek41 => |hp| gguf.gguf_arch.deepseek41.isDsparkArch(name) and
            if (gguf.gguf_arch.deepseek41.Dspark.read(h, hp)) |_| true else |_| false,
        else => false,
    };
}

/// Fill `out` from the GGUF. False when `model_dir` is not ours.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, out: *std.StringHashMap(mlx.mlx_array)) !bool {
    if (comptime @import("builtin").os.tag == .windows) return false; // Metal-only engine
    const path = servablePath(io, allocator, model_dir) orelse return false;
    defer allocator.free(path);
    var f = try gguf.gguf.File.open(allocator, path);
    defer f.deinit();
    try gguf.weights.load(allocator, &f, out, mlx.gpuStream());
    if (mtpHead(io, allocator, &f, path)) |head| {
        defer allocator.free(head);
        var h = try gguf.gguf.File.open(allocator, head);
        defer h.deinit();
        switch (try gguf.gguf_arch.Arch.read(&f)) {
            .deepseek41 => |hp| try gguf.weights.loadDspark(allocator, &h, hp, out, mlx.gpuStream()),
            else => try gguf.weights.loadMtp(allocator, &h, out, mlx.gpuStream()),
        }
        log.info("[gguf] mlx engine: draft {s}\n", .{std.fs.path.basename(head)});
    }
    try gguf.weights.warmKernels(allocator, out, mlx.gpuStream());
    log.info("[gguf] mlx engine: {d} tensors from {s}\n", .{ f.tensors.count(), path });
    return true;
}

/// The vision tower GGUF beside the set `model_dir` serves into `out` (the engine's `load_vision`). False when there
/// is none, or `model_dir` is not ours.
pub fn loadVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, out: *std.StringHashMap(mlx.mlx_array)) !bool {
    if (comptime @import("builtin").os.tag == .windows) return false; // Metal-only engine
    const path = servablePath(io, allocator, model_dir) orelse return false;
    defer allocator.free(path);
    var f = try gguf.gguf.File.open(allocator, path);
    defer f.deinit();
    const tower = visionTower(io, allocator, &f, path) orelse return false;
    defer allocator.free(tower);
    var v = try gguf.gguf.File.open(allocator, tower);
    defer v.deinit();
    try gguf.weights.loadVision(allocator, &v, (try gguf.gguf_arch.Arch.read(&f)).deepseek41, out, mlx.gpuStream());
    log.info("[gguf] mlx engine: vision {s}\n", .{std.fs.path.basename(tower)});
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
        try std.testing.expect(!try loadVision(io, allocator, dir, &map));
    }
}

test "a DeepSeek-V4.1 set drafts with either converter's DSpark bundle, never another arch's or another trunk's stages" {
    const ds41 = gguf.gguf_arch.deepseek41;
    var t = try ds41.testFileDs4(ds41.release_config);
    defer t.deinit();
    try t.kv.put(t.arena.allocator(), "general.architecture", .{ .string = ds41.arch_name });
    const trunk = try gguf.gguf_arch.Arch.read(&t);
    var unsloth = try ds41.testDsparkFile();
    defer unsloth.deinit();
    var ds4 = try ds41.testDsparkFileDs4();
    defer ds4.deinit();
    try std.testing.expect(draftsWith(trunk, &unsloth));
    try std.testing.expect(draftsWith(trunk, &ds4));
    try ds4.tensors.put(ds4.arena.allocator(), "mtp.1.ffn_gate_exps.weight", .{ .dims = .{ 5120, 2304, 64, 1 }, .n_dims = 3, .ty = .mxfp4, .data = &.{} });
    try std.testing.expect(!draftsWith(trunk, &ds4));
    try ds4.kv.put(ds4.arena.allocator(), "general.architecture", .{ .string = "qwen4exp" });
    try std.testing.expect(!draftsWith(trunk, &ds4));
}

test "a DeepSeek-V4.1 set serves images with the vision GGUF beside it from its own checkpoint, never another's" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    const ds41 = gguf.gguf_arch.deepseek41;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const set = try std.fmt.allocPrint(a, "{s}/V4.1-IQ2.gguf", .{root});
    defer a.free(set);
    var f = try ds41.testFileDs4(ds41.release_config);
    defer f.deinit();
    try f.kv.put(f.arena.allocator(), "general.architecture", .{ .string = ds41.arch_name });
    try f.kv.put(f.arena.allocator(), "general.source.revision", .{ .string = "1bae0e1088993b9b82de08f59de5be5a34e42170" });
    var v = try ds41.testVisionFileDs4();
    defer v.deinit();
    for ([_][]const u8{ "1bae0e1088993b9b82de08f59de5be5a34e42170", "df42c109f1defefcbfcedbe7d905718a12266e40" }, [_]bool{ true, false }) |rev, pairs| {
        try v.kv.put(v.arena.allocator(), "general.source.revision", .{ .string = rev });
        var hdr: std.ArrayList(u8) = .empty;
        defer hdr.deinit(a);
        try hdr.appendSlice(a, "GGUF" ++ std.mem.toBytes(@as(u32, 3)) ++ std.mem.toBytes(@as(u64, 0)) ++ std.mem.toBytes(@as(u64, v.kv.count())));
        var it = v.kv.iterator();
        while (it.next()) |e| {
            try hdr.appendSlice(a, &std.mem.toBytes(@as(u64, e.key_ptr.len)));
            try hdr.appendSlice(a, e.key_ptr.*);
            switch (e.value_ptr.*) {
                .string => |str| {
                    try hdr.appendSlice(a, &(std.mem.toBytes(@as(u32, 8)) ++ std.mem.toBytes(@as(u64, str.len))));
                    try hdr.appendSlice(a, str);
                },
                .uint => |n| try hdr.appendSlice(a, &(std.mem.toBytes(@as(u32, 4)) ++ std.mem.toBytes(@as(u32, @intCast(n))))),
                .float => |x| try hdr.appendSlice(a, &(std.mem.toBytes(@as(u32, 6)) ++ std.mem.toBytes(@as(f32, @floatCast(x))))),
                else => unreachable,
            }
        }
        try tmp.dir.writeFile(io, .{ .sub_path = "V4.1-1017-Vision.gguf", .data = hdr.items });
        const json = try configJson(io, a, &f, set);
        defer a.free(json);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer parsed.deinit();
        const vc = parsed.value.object.get("vision_config");
        try std.testing.expectEqual(pairs, vc != null);
        try std.testing.expectEqual(pairs, parsed.value.object.get("image_token_id") != null);
        if (vc) |c| try std.testing.expectEqual(@as(i64, 2816), c.object.get("intermediate_size").?.integer);
    }
}

test {
    _ = gguf;
    _ = gguf_files;
}
