//! Which real packs a served type's plugin claims: a flipped plugin that declines a released pack hands it back to
//! the legacy path silently, and synthetic proofs never see it. Reads config.json and its sidecars only (the
//! parse's own reads: safetensors headers at most), never weights or MLX.

const std = @import("std");
const plugin_config = @import("config.zig");
const served = @import("served.zig");
const sdk_plugin = @import("../../../sdk/plugin.zig");
const registry = @import("../../../plugins/registry.zig");

/// One pack's verdict: the plugin serves it (`why` null) or not, and why.
pub const Row = struct { model_type: []const u8, why: ?[]const u8 };

/// The verdict on the pack at `model_dir`; `model_type` and `why` are static or arena-owned.
pub fn auditPack(io: std.Io, arena: std.mem.Allocator, model_dir: []const u8, config_json: []const u8) Row {
    const model_type = sdk_plugin.modelTypeOf(arena, config_json) catch |e| return .{ .model_type = "?", .why = @errorName(e) };
    const out = plugin_config.resolve(io, std.heap.page_allocator, model_dir, config_json) catch |e| return .{ .model_type = model_type, .why = @errorName(e) };
    switch (out) {
        .bound => |b| {
            b.destroy(std.heap.page_allocator);
            return .{ .model_type = model_type, .why = null };
        },
        .declined => |why| return .{ .model_type = model_type, .why = why },
    }
}

const Tally = struct { served: u32 = 0, legacy: u32 = 0 };

fn walk(io: std.Io, arena: std.mem.Allocator, path: []const u8, depth: u8, tally: *std.StringArrayHashMapUnmanaged(Tally)) !void {
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    if (dir.readFileAlloc(io, "config.json", arena, .limited(16 * 1024 * 1024))) |json| {
        const row = auditPack(io, arena, path, json);
        std.debug.print("{s:<26} {s:<4} {s:<48} {s}\n", .{ row.model_type, if (row.why == null) "yes" else "no", row.why orelse "-", path });
        if (!served.serves(row.model_type)) return;
        const gop = try tally.getOrPut(arena, row.model_type);
        if (!gop.found_existing) gop.value_ptr.* = .{};
        if (row.why == null) gop.value_ptr.served += 1 else gop.value_ptr.legacy += 1;
        return;
    } else |_| {}
    if (depth == 0) return;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| {
        if (e.name[0] == '.' or (e.kind != .directory and e.kind != .sym_link)) continue;
        try walk(io, arena, try std.fs.path.join(arena, &.{ path, e.name }), depth - 1, tally);
    }
}

test "served audit: every pack under MLX_SERVE_SERVED_AUDIT_DIRS, its type, plugin or legacy, and why" {
    const roots = std.c.getenv("MLX_SERVE_SERVED_AUDIT_DIRS") orelse return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tally: std.StringArrayHashMapUnmanaged(Tally) = .empty;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(roots), ':');
    while (it.next()) |root| try walk(io, arena, root, 3, &tally);
    std.debug.print("\nserved type                 plugin legacy\n", .{});
    for (tally.keys(), tally.values()) |t, n| std.debug.print("{s:<26} {d:>7} {d:>6}\n", .{ t, n.served, n.legacy });
}

/// Fixture packs of a served type that the legacy path keeps (`why` the plugin's decline), each with the reason
/// legacy also refuses it.
const legacy_fixtures = [_]struct { name: []const u8, why: []const u8, legacy: []const u8 }{
    .{ .name = "gemma4-text.json", .why = "gemma: a draft model's config", .legacy = "a DSpark sidecar: discovery classifies the DFlash contract as a drafter, its load is UnsupportedArch" },
    .{ .name = "gemma4-unified-text.json", .why = "gemma: a draft model's config", .legacy = "an EAGLE-3 head: one draft layer, the trunk's layers are MissingWeight" },
    .{ .name = "nemotron_h_super_drafter.json", .why = "qwen3: a quantization_config naming a quant_method", .legacy = "a LiLiCorr DFlash sidecar: discovery classifies it as a drafter" },
    .{ .name = "mistral.json", .why = "llama: a quantization_config naming a quant_method", .legacy = "bitsandbytes nf4 weights: no MLX scales, the load is MissingWeight" },
    .{ .name = "nemotron-h.json", .why = "nemotron_h: a quantization_config block", .legacy = "compressed-tensors FP8 (weight_scale, not MLX scales)" },
};

test "served audit: every fixture config of a served type is claimed, or its decline is listed" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var failed = false;
    for (registry.model_configs) |f| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = f.bytes });
        var buf: [512]u8 = undefined;
        const dir = buf[0..try tmp.dir.realPath(io, &buf)];
        const row = auditPack(io, arena, dir, f.bytes);
        if (!served.serves(row.model_type)) continue;
        const why = row.why orelse continue;
        for (legacy_fixtures) |l| {
            if (std.mem.eql(u8, l.name, f.name) and std.mem.eql(u8, l.why, why)) break;
        } else {
            std.debug.print("{s} ({s}): declined, {s}\n", .{ f.name, row.model_type, why });
            failed = true;
        }
    }
    try std.testing.expect(!failed);
}
