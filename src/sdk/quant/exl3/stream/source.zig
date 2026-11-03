//! The EXL3 expert stream as the expert_source kind's registry entry (G6): discovery and `/props` ask it; the arch
//! that streams with it binds `expert_stream.Stream` at comptime, so nothing per layer crosses this table.

const std = @import("std");
const sdk = @import("sdk");
const expert_stream = @import("../../../weights/streamed/expert/stream.zig");

pub const name = "exl3-stream";
pub const caps = expert_stream.source_caps;
pub const uses_reader = expert_stream.uses_reader;

/// A model directory holding an expert bank's v2 manifest, by presence. The bank's format, quantization and geometry
/// are checked when it opens (`expert_bank.Bank.open`, the quant's claim at load).
/// glm5_next's EXL3 pack holds the same manifest; that arch streams its bank through its own source (the stream's bank
/// module over this reader), so the DeepSeek entry declines it.
pub fn claims(p: *const sdk.ConfigPeek) ?sdk.Priority {
    if (p.model_dir.len == 0) return null;
    if (p.modelType()) |t| if (std.mem.eql(u8, t, "glm5_next")) return null;
    var buf: [4096]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&buf, "{s}/expert-manifest-v2.json", .{p.model_dir}, 0) catch return null;
    return if (std.c.access(path.ptr, 0) == 0) .native else null;
}

const testing = std.testing;

test "dsv41 plugin: the EXL3 source claims a model directory that holds an expert bank, and nothing else" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = rbuf[0..try tmp.dir.realPath(std.testing.io, &rbuf)];
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const peek = try sdk.ConfigPeek.parse(arena.allocator(), root, "{\"model_type\":\"deepseek_v41\"}");
    try testing.expectEqual(@as(?sdk.Priority, null), claims(&peek));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "expert-manifest-v2.json", .data = "{}" });
    try testing.expectEqual(@as(?sdk.Priority, .native), claims(&peek));
    // glm5_next's EXL3 pack: the same manifest, another arch's source.
    const glm = try sdk.ConfigPeek.parse(arena.allocator(), root, "{\"model_type\":\"glm5_next\"}");
    try testing.expectEqual(@as(?sdk.Priority, null), claims(&glm));
}
